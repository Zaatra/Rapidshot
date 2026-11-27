Texture2D<uint> Source : register(t0);
Buffer<uint> Selection : register(t1);
RWBuffer<uint> Result : register(u0);

cbuffer Geometry : register(b0)
{
    uint Width;
    uint Height;
    uint Columns;
    uint TileSize;
};

groupshared uint2 Stripes[256];

uint2 Rotate(uint2 value, uint2 bits)
{
    return (value << bits) | (value >> (32 - bits));
}

// Two seeded 32-bit lanes, with ordered mixing within and between stripes.
// This benchmark fingerprint is probabilistic and does not implement a standard digest.
uint2 Mix(uint2 state, uint2 word)
{
    uint2 key = Rotate(word * uint2(0xcc9e2d51, 0x9e3779b1), uint2(15, 17));
    state ^= key * uint2(0x1b873593, 0x85ebca77);
    return Rotate(state, uint2(13, 19)) * 5 + uint2(0xe6546b64, 0x165667b1);
}

void ProcessTile(uint tile, uint2 thread, uint index, bool deliver, bool hash, uint hash_offset)
{
    uint2 origin = uint2(tile % Columns, tile / Columns) * TileSize;
    uint2 end = min(origin + TileSize, uint2(Width, Height));
    uint2 state = uint2(0x243f6a88, 0xb7e15162) ^ index;
    for (uint y = origin.y + thread.y; y < end.y; y += 16)
    {
        for (uint x = origin.x + thread.x; x < end.x; x += 16)
        {
            uint word = Source.Load(int3(x, y, 0));
            if (deliver)
                Result[y * Width + x] = word;
            if (hash)
                state = Mix(state, uint2(word, word));
        }
    }
    if (!hash)
        return;

    Stripes[index] = state;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = 128; stride > 0; stride >>= 1)
    {
        if (index < stride)
            Stripes[index] = Mix(Stripes[index], Stripes[index + stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (index == 0)
    {
        Result[hash_offset] = Stripes[0].x;
        Result[hash_offset + 1] = Stripes[0].y;
    }
}

[numthreads(16, 16, 1)]
void Fingerprint(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID, uint index : SV_GroupIndex)
{
    ProcessTile(Selection[group.x], thread.xy, index, false, true, group.x * 2);
}

[numthreads(16, 16, 1)]
void CopyPixels(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID, uint index : SV_GroupIndex)
{
    ProcessTile(group.x, thread.xy, index, true, false, 0);
}

[numthreads(16, 16, 1)]
void Fused(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID, uint index : SV_GroupIndex)
{
    // Selection is indexed by tile for fused delivery. Zero indicates that the
    // tile's pixels must be delivered without computing a fingerprint.
    uint ordinal = Selection[group.x];
    ProcessTile(group.x, thread.xy, index, true, ordinal != 0, Width * Height + (ordinal - 1) * 2);
}
