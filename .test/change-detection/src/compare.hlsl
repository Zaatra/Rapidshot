Texture2D<uint> Previous : register(t0);
Texture2D<uint> Current : register(t1);
Buffer<uint> Candidates : register(t2);
RWBuffer<uint> Changed : register(u0);

cbuffer Geometry : register(b0)
{
    uint Width;
    uint Height;
    uint Columns;
    uint TileSize;
};

groupshared uint Differences[256];

[numthreads(16, 16, 1)]
void Compare(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID, uint index : SV_GroupIndex)
{
    uint tile = Candidates[group.x];
    uint2 origin = uint2(tile % Columns, tile / Columns) * TileSize;
    uint2 end = min(origin + TileSize, uint2(Width, Height));
    uint difference = 0;

    // Each lane checks a strided subset of one candidate tile, including partial
    // tiles at the screen edge. XOR compares all BGRA bytes without a hash.
    for (uint y = origin.y + thread.y; y < end.y; y += 16)
    {
        for (uint x = origin.x + thread.x; x < end.x; x += 16)
        {
            int3 position = int3(x, y, 0);
            difference |= Previous.Load(position) ^ Current.Load(position);
        }
    }
    Differences[index] = difference;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = 128; stride > 0; stride >>= 1)
    {
        if (index < stride)
        {
            Differences[index] |= Differences[index + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (index == 0)
    {
        Changed[group.x] = Differences[0] != 0 ? 1 : 0;
    }
}
