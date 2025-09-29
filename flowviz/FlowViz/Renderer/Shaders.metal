#include <metal_stdlib>
using namespace metal;

struct Particle {
    float2 position;
    float2 velocity;
    float life;
    float maxLife;
};

struct Uniforms {
    float time;
    float deltaTime;
    float2 screenSize;
    float aspectRatio;
    float flowSpeed;
    float particleSize;
};

struct VertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
    float life;
    float4 color;
};

// MARK: - Vertex Shader

vertex VertexOut particle_vertex(uint vertexID [[vertex_id]],
                                constant Particle* particles [[buffer(0)]],
                                constant Uniforms& uniforms [[buffer(1)]]) {
    Particle particle = particles[vertexID];
    
    VertexOut out;
    
    // Convert normalized coordinates to clip space
    float2 clipPos = particle.position * 2.0 - 1.0;
    clipPos.y = -clipPos.y; // Flip Y coordinate
    out.position = float4(clipPos, 0.0, 1.0);
    
    // Calculate particle properties
    float lifeRatio = particle.life / particle.maxLife;
    out.life = lifeRatio;
    out.pointSize = uniforms.particleSize * (0.5 + lifeRatio * 0.5);
    
    // Color based on velocity magnitude and life
    float speed = length(particle.velocity);
    float3 baseColor = float3(0.2, 0.8, 1.0); // Cyan base
    float3 fastColor = float3(1.0, 0.4, 0.8);  // Pink for fast particles
    
    float speedFactor = saturate(speed * 10.0);
    float3 color = mix(baseColor, fastColor, speedFactor);
    
    out.color = float4(color, lifeRatio * 0.8);
    
    return out;
}

// MARK: - Fragment Shader

fragment float4 particle_fragment(VertexOut in [[stage_in]],
                                 float2 pointCoord [[point_coord]],
                                 constant Uniforms& uniforms [[buffer(0)]]) {
    // Create circular particle with soft edges
    float2 center = float2(0.5, 0.5);
    float distance = length(pointCoord - center);
    
    // Soft circular falloff
    float alpha = 1.0 - smoothstep(0.3, 0.5, distance);
    alpha *= in.color.a;
    
    // Add some glow effect
    float glow = 1.0 - smoothstep(0.0, 0.8, distance);
    glow *= 0.3;
    
    float4 finalColor = in.color;
    finalColor.a = alpha + glow;
    
    return finalColor;
}

// MARK: - Compute Shader for Particle Updates

kernel void update_particles(uint index [[thread_position_in_grid]],
                           device Particle* particles [[buffer(0)]],
                           constant float2* velocityField [[buffer(1)]],
                           constant Uniforms& uniforms [[buffer(2)]]) {
    
    if (index >= 10000) return; // Max particle count
    
    Particle particle = particles[index];
    
    // Sample velocity from grid (bilinear interpolation)
    float2 gridPos = particle.position * 127.0; // 128x128 grid, 0-127 indices
    int2 gridIndex = int2(gridPos);
    float2 frac = fract(gridPos);
    
    // Clamp to grid bounds
    gridIndex = clamp(gridIndex, int2(0), int2(126));
    
    // Sample four neighboring grid points
    float2 v00 = velocityField[gridIndex.y * 128 + gridIndex.x];
    float2 v10 = velocityField[gridIndex.y * 128 + (gridIndex.x + 1)];
    float2 v01 = velocityField[(gridIndex.y + 1) * 128 + gridIndex.x];
    float2 v11 = velocityField[(gridIndex.y + 1) * 128 + (gridIndex.x + 1)];
    
    // Bilinear interpolation
    float2 v0 = mix(v00, v10, frac.x);
    float2 v1 = mix(v01, v11, frac.x);
    float2 sampledVelocity = mix(v0, v1, frac.y);
    
    // Update particle velocity (with some inertia)
    particle.velocity = mix(particle.velocity, sampledVelocity * uniforms.flowSpeed, 0.1);
    
    // Update position using Euler integration
    particle.position += particle.velocity * uniforms.deltaTime;
    
    // Boundary conditions (wrap around)
    if (particle.position.x < 0.0) particle.position.x = 1.0;
    if (particle.position.x > 1.0) particle.position.x = 0.0;
    if (particle.position.y < 0.0) particle.position.y = 1.0;
    if (particle.position.y > 1.0) particle.position.y = 0.0;
    
    // Update life
    particle.life -= uniforms.deltaTime;
    
    // Respawn particle if dead
    if (particle.life <= 0.0) {
        particle.position = float2(frand(index * 2), frand(index * 2 + 1));
        particle.velocity = float2(0.0);
        particle.life = particle.maxLife;
    }
    
    particles[index] = particle;
}

// Simple pseudo-random function
float frand(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) / float(0xffffffffu);
}
