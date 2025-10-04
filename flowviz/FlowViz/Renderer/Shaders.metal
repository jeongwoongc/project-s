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
    
    // Dynamic particle size based on life and speed
    float speed = length(particle.velocity);
    float sizeMultiplier = 1.0 + speed * 2.0; // Faster particles are bigger
    out.pointSize = uniforms.particleSize * (0.3 + lifeRatio * 0.7) * sizeMultiplier;
    
    // Enhanced color palette with smooth transitions - vibrant cool tones
    float speedFactor = saturate(speed * 8.0);
    
    // Create a rainbow gradient based on speed and position
    float hueShift = speedFactor + uniforms.time * 0.1 + particle.position.x * 0.3;
    hueShift = fract(hueShift);
    
    float3 slowColor = float3(0.1, 0.4, 1.0);      // Deep blue
    float3 medColor = float3(0.2, 0.9, 1.0);       // Bright cyan
    float3 fastColor = float3(0.9, 0.3, 1.0);      // Vibrant purple/magenta
    float3 ultraFastColor = float3(0.5, 1.0, 1.0); // Electric cyan
    
    float3 color;
    if (speedFactor < 0.33) {
        color = mix(slowColor, medColor, speedFactor * 3.0);
    } else if (speedFactor < 0.66) {
        color = mix(medColor, fastColor, (speedFactor - 0.33) * 3.0);
    } else {
        color = mix(fastColor, ultraFastColor, (speedFactor - 0.66) * 3.0);
    }
    
    // Add subtle pulsing based on time
    float pulse = 0.5 + 0.5 * sin(uniforms.time * 2.0 + float(vertexID) * 0.01);
    color = mix(color, color * 1.3, pulse * 0.2);
    
    out.color = float4(color, lifeRatio * 0.9);
    
    return out;
}

// MARK: - Fragment Shader

fragment float4 particle_fragment(VertexOut in [[stage_in]],
                                 float2 pointCoord [[point_coord]],
                                 constant Uniforms& uniforms [[buffer(0)]]) {
    // Create circular particle with enhanced glow
    float2 center = float2(0.5, 0.5);
    float dist = length(pointCoord - center);
    
    // Multi-layered particle with core and glow
    // Bright core
    float core = 1.0 - smoothstep(0.0, 0.2, dist);
    
    // Middle layer
    float middle = 1.0 - smoothstep(0.2, 0.4, dist);
    
    // Outer glow
    float glow = 1.0 - smoothstep(0.0, 0.5, dist);
    glow = pow(glow, 2.0); // Exponential falloff for softer glow
    
    // Combine layers
    float alpha = core * 1.0 + middle * 0.7 + glow * 0.4;
    alpha *= in.color.a;
    
    // Add shimmer effect
    float shimmer = 1.0 + 0.3 * sin(uniforms.time * 5.0 + dist * 10.0);
    
    // Enhanced color with brightness boost in center
    float3 finalColor = in.color.rgb;
    finalColor = mix(finalColor, finalColor * 1.5, core * 0.5); // Brighter core
    finalColor *= shimmer;
    
    // Add bloom effect for very bright particles
    float brightness = length(finalColor);
    float bloom = smoothstep(1.0, 2.0, brightness) * glow * 0.5;
    
    return float4(finalColor, alpha + bloom);
}

// Forward declaration for frand utility
float frand(uint seed);

// MARK: - Compute Shader for Particle Updates

kernel void update_particles(uint index [[thread_position_in_grid]],
                           device Particle* particles [[buffer(0)]],
                           constant float2* velocityField [[buffer(1)]],
                           constant Uniforms& uniforms [[buffer(2)]],
                           constant uint& particleCount [[buffer(3)]]) {
    
    if (index >= particleCount) return; // Dynamic particle count
    
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
    
    // Add subtle turbulence for more organic motion
    float turbulence = sin(uniforms.time * 2.0 + float(index) * 0.1) * 0.002;
    float2 turbulenceVec = float2(
        cos(uniforms.time + particle.position.y * 10.0),
        sin(uniforms.time + particle.position.x * 10.0)
    ) * turbulence;
    
    // Update particle velocity with inertia and turbulence
    float2 targetVelocity = sampledVelocity * uniforms.flowSpeed + turbulenceVec;
    particle.velocity = mix(particle.velocity, targetVelocity, 0.15);
    
    // Update position using improved Euler integration
    particle.position += particle.velocity * uniforms.deltaTime;
    
    // Smooth boundary conditions with fade and wrap
    float fadeMargin = 0.02;
    if (particle.position.x < -fadeMargin) particle.position.x = 1.0 + fadeMargin;
    if (particle.position.x > 1.0 + fadeMargin) particle.position.x = -fadeMargin;
    if (particle.position.y < -fadeMargin) particle.position.y = 1.0 + fadeMargin;
    if (particle.position.y > 1.0 + fadeMargin) particle.position.y = -fadeMargin;
    
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

