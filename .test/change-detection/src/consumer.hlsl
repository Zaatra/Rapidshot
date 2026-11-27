Buffer<uint> Candidates : register(t0);
Buffer<uint> Flags : register(t1);
Texture2D<uint> Current : register(t2);
Buffer<uint> WorkIn : register(t3);
RWBuffer<uint> WorkOut : register(u0);
RWByteAddressBuffer Args : register(u1);
RWBuffer<uint> Scores : register(u2);

cbuffer Parameters : register(b0)
{
    uint Width;
    uint Height;
    uint Columns;
    uint TileSize;
    uint CandidateCount;
};

[numthreads(1, 1, 1)]
void Compact()
{
    uint count = 0;
    for (uint i = 0; i < CandidateCount; ++i)
    {
        if (Flags[i] != 0)
        {
            WorkOut[count++] = Candidates[i];
        }
    }
    Args.Store3(0, uint3(count, 1, 1));
}

groupshared uint PartialScores[256];

[numthreads(16, 16, 1)]
void Score(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID, uint index : SV_GroupIndex)
{
    uint tile = WorkIn[group.x];
    uint2 origin = uint2(tile % Columns, tile / Columns) * TileSize;
    uint2 end = min(origin + TileSize, uint2(Width, Height));
    uint sum = 0;
    for (uint y = origin.y + thread.y; y < end.y; y += 16)
    {
        for (uint x = origin.x + thread.x; x < end.x; x += 16)
        {
            uint pixel = Current.Load(int3(x, y, 0));
            sum += (pixel & 255) * 3 + ((pixel >> 8) & 255) * 6 + ((pixel >> 16) & 255);
        }
    }
    PartialScores[index] = sum;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = 128; stride > 0; stride >>= 1)
    {
        if (index < stride)
            PartialScores[index] += PartialScores[index + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (index == 0)
        Scores[tile] = PartialScores[0];
}
