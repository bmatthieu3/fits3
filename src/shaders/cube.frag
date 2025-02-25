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

const float fov = 30.0 * 3.14 / 180.0;
const float camera_near = 1.0;
const float dmin = -2.451346722E-03;
const float dmax = 1.179221552E-02;
void main() {
    // Retrieve the position from the texture
    //vec3 pos_ws = normalize(pos_xyz);
    // Rotate it
    //vec3 rotated_p = vec3(rot * vec4(pos_ws, 1.0));
    // camera turning
    vec3 cam_origin = 2.0 * lonlat2xyz(origin.x, origin.y);

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
    //vec3 r = normalize(p_cam - cam_origin);
    // orthographic perspective
    vec3 r = cam_dir;

    // we define our cube as 2 bounds vertices, l and h
    vec3 l = vec3(-0.5, -0.5, -0.5);
    vec3 h = vec3(0.5, 0.5, 0.5);

    vec3 t_low = (l - p_cam) / r;
    vec3 t_high = (h - p_cam) / r;

    vec3 t_close = min(t_low, t_high);
    vec3 t_far = max(t_low, t_high);

    float t_c = max(t_close.x, max(t_close.y, t_close.z));
    float t_f = min(t_far.x, min(t_far.y, t_far.z));

    if (t_f < t_c) {
        f_color = vec4(vec3(0.0), 1.0);
    } else {
        float intensity = 0.0;
        float step = 0.01;
        float num_sampling = (t_f - t_c) / step;

        float scale = cut.x;
        float off = cut.y;
        for (float t = t_c; t < t_f; t += step) {
            // absolute sampling point
            vec3 p = p_cam + r * t;
            // scaled to the origin of the cube
            vec3 v = p - l;
            // v lies in [0; 1]^3

            float t_val = to_l_endian(texture(sampler3D(t_map, s_map), v).r);

            float t_val_norm = (t_val - dmin) / (dmax - dmin);

            intensity += t_val_norm;
        }
        intensity /= num_sampling;
        intensity = intensity * scale + off;

        f_color = colormap(intensity);
    }
}
 