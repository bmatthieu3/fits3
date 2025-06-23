// shader.frag
#version 440

layout(location=0) in vec2 ndc;
layout(location=0) out vec4 f_color;

layout(set = 0, binding = 0) uniform texture3D t_map;
layout(set = 0, binding = 1) uniform sampler s_map;
layout(set = 0, binding = 2)
uniform RotationMatrix {
    mat4 rot;
};
layout(set = 0, binding = 4)
uniform Time {
    vec4 time;
};
layout(set = 0, binding = 5)
uniform Origin {
    vec4 origin;
};
layout(set = 0, binding = 6)
uniform Cut {
    vec4 cut;
};

vec3 lonlat2xyz(float lon, float lat) {
    float lat_s = sin(lat);
    float lat_c = cos(lat);
    float lon_s = sin(lon);
    float lon_c = cos(lon);

    return vec3(lat_c * lon_s, lat_s, lat_c * lon_c);
}
float colormap_red(float x) {
    if (x < 0.7) {
        return 4.0 * x - 1.5;
    } else {
        return -4.0 * x + 4.5;
    }
}

float colormap_green(float x) {
    if (x < 0.5) {
        return 4.0 * x - 0.5;
    } else {
        return -4.0 * x + 3.5;
    }
}

float colormap_blue(float x) {
    if (x < 0.3) {
       return 4.0 * x + 0.5;
    } else {
       return -4.0 * x + 2.5;
    }
}
/*
vec3 colormap_viridis(float t) {
    // Clamp input to [0,1]
    t = clamp(t, 0.0, 1.0);

    // Coefficients from the original viridis colormap (Matplotlib)
    const vec3 c0 = vec3(0.280, 0.165, 0.476);
    const vec3 c1 = vec3(0.110, 0.573, 0.664);
    const vec3 c2 = vec3(0.478, 0.821, 0.318);

    // Interpolation logic
    if (t < 0.5) {
        float f = smoothstep(0.0, 0.5, t);
        return mix(c0, c1, f);
    } else {
        float f = smoothstep(0.5, 1.0, t);
        return mix(c1, c2, f);
    }
}*/

vec3 colormap_viridis(float t) {
    vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
    vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
    vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}

vec3 colormap_turbo(in float x) {
    const vec4 kRedVec4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kBlueVec4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kRedVec2 = vec2(-152.94239396, 59.28637943);
    const vec2 kGreenVec2 = vec2(4.27729857, 2.82956604);
    const vec2 kBlueVec2 = vec2(-89.90310912, 27.34824973);
  
    x = clamp(x,0.0,1.0);
    vec4 v4 = vec4( 1.0, x, x * x, x * x * x);
    vec2 v2 = v4.zw * v4.z;
    return vec3(
        dot(v4, kRedVec4)   + dot(v2, kRedVec2),
        dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
        dot(v4, kBlueVec4)  + dot(v2, kBlueVec2)
    );
}

vec4 colormap(float x) {
    float r = clamp(colormap_red(x), 0.0, 1.0);
    float g = clamp(colormap_green(x), 0.0, 1.0);
    float b = clamp(colormap_blue(x), 0.0, 1.0);
    return vec4(r, g, b, 1.0);
}

float to_l_endian(float x) {
    uint y = floatBitsToUint(x);

    uint a = y & 0xff;
    uint b = (y >> 8) & 0xff;
    uint c = (y >> 16) & 0xff;
    uint d = y >> 24;

    uint w = (a << 24) | (b << 16) | (c << 8) | d;

    return uintBitsToFloat(w);
}

// Parameters:
//   x - input intensity (usually normalized to [0,1])
//   scale - scaling factor to control the stretch strength
//   nonlinearity - controls how nonlinear the stretch is (typically > 0)
//
// Returns:
//   A value between 0 and 1 after applying the asinh stretch
float asinhStretch(float x, float scale, float nonlinearity) {
    return asinh(scale * x) / asinh(scale * nonlinearity);
}

const float fov = 0.523333;
const float camera_near = 1.0;
//const float dmin = -2.451346722E-03;
//const float dmax = 1.179221552E-02;
const float dmin = -1.0;
const float dmax = 2.0;
// we define our cube as 2 bounds vertices, l and h
const vec3 l = vec3(-0.5, -0.5, -0.5);
const vec3 h = vec3(0.5, 0.5, 0.5);
void main() {
    vec3 cam_origin = 1.5 * lonlat2xyz(origin.x, origin.y);

    // vector from camera origin to the look
    vec3 cam_dir = normalize(-cam_origin);
    // origin of the screen in world space
    vec3 o_cam = cam_origin + cam_dir * camera_near;

    // find a vector belonging to the plane of screen oriented with y
    vec3 ox = normalize(vec3(cam_dir.z, 0.0, -cam_dir.x));
    vec3 oy = -cross(ox, cam_dir);

    vec3 p_cam = o_cam + ox * ndc.x + oy * ndc.y;

    // vector director from the cam origin to the pixel on screen
    // traditional perspective director vector
    // orthographic perspective
    vec3 r = cam_dir;
    // perspective
    //vec3 r = normalize(p_cam - cam_origin);

    vec3 t_low = (l - p_cam) / r;
    vec3 t_high = (h - p_cam) / r;

    vec3 t_close = min(t_low, t_high);
    vec3 t_far = max(t_low, t_high);

    float t_c = max(t_close.x, max(t_close.y, t_close.z));
    float t_f = min(t_far.x, min(t_far.y, t_far.z));

    if (t_f < t_c) {
        discard;
    }

    float intensity = 0.0;
    float step = 0.01;
    int num_sampling = int((t_f - t_c) / step);
    /*
    int num_sampling = min(int((t_f - t_c) / step), 50);
    if (num_sampling == 50) {
        step = (t_f - t_c) / float(num_sampling);
    }*/
    //float num_sampling = 100.0;
    //float step = max((t_f - t_c) / num_sampling, 0.0005);

    vec3 dr = r * step;

    //float random = fract(sin(gl_FragCoord.x * 12.9898 + gl_FragCoord.y * 78.233) * 43758.5453);
    float random = 0.0;
    float t_s = t_c + step * random;
    // absolute sampling point
    // scaled to the origin of the cube
    vec3 p = p_cam + r * t_s - l;
    float alpha = 1.0;
    for (int i = 0; i < num_sampling; i++) {
        intensity += to_l_endian(texture(sampler3D(t_map, s_map), p).r);
        /*alpha *= 0.97;
        if (alpha < 1e-3) {
            break;
        }*/
        p += dr;
    }
    intensity /= num_sampling;
    intensity = ((intensity - dmin) / (dmax - dmin)) * cut.x + cut.y;

    //intensity = asinhStretch(intensity, 2.0, 1.0);

    //f_color = vec4(colormap_turbo(intensity), 1.0);
    f_color = colormap(intensity);
    //f_color = vec4(intensity);
}
 