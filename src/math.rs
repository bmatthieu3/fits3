use cgmath::BaseFloat;

pub(crate) type Vec2<T> = cgmath::Vector2<T>;
pub(crate) type Vec3<T> = cgmath::Vector3<T>;
pub(crate) type Vec4<T> = cgmath::Vector4<T>;


pub(crate) type Mat4<T> = cgmath::Matrix4<T>;
pub(crate) type Mat3<T> = cgmath::Matrix3<T>;

#[inline]
pub(crate) fn xyz_to_radec(v: &cgmath::Vector3<f32>) -> (f32, f32) {
    let lon = v.x.atan2(v.z);
    let lat = v.y.atan2((v.x * v.x + v.z * v.z).sqrt());

    (lon, lat)
}

#[inline]
pub(crate) fn xyzw_to_radec(v: &cgmath::Vector4<f32>) -> (f32, f32) {
    let lon = v.x.atan2(v.z);
    let lat = v.y.atan2((v.x * v.x + v.z * v.z).sqrt());

    (lon, lat)
}

#[inline]
pub(crate) fn radec_to_xyzw(theta: f32, delta: f32) -> Vec4<f32> {
    let (d_s, d_c) = delta.to_radians().sin_cos();
    let (t_s, t_c) = theta.to_radians().sin_cos();

    Vec4::<f32>::new(d_c * t_s, d_s, d_c * t_c, 1.0)
}

#[inline]
pub(crate) fn radec_to_xyz(theta: f32, delta: f32) -> Vec3<f32> {
    let (d_s, d_c) = delta.to_radians().sin_cos();
    let (t_s, t_c) = theta.to_radians().sin_cos();

    Vec3::<f32>::new(d_c * t_s, d_s, d_c * t_c)
}

#[inline]
pub(crate) fn asinc_positive(mut x: f32) -> f32 {
    assert!(x >= 0.0);
    if x > 1.0e-4 {
        x.asin() / x
    } else {
        // If a is mall, use Taylor expension of asin(a) / a
        // a = 1e-4 => a^4 = 1.e-16
        x *= x;

        1.0 + x * (1.0 + x * 9.0 / 20.0) / 6.0
    }
}

#[inline]
pub(crate) fn sinc_positive(mut x: f32) -> f32 {
    assert!(x >= 0.0);
    if x > 1.0e-4 {
        x.sin() / x
    } else {
        // If a is mall, use Taylor expension of asin(a) / a
        // a = 1e-4 => a^4 = 1.e-16
        x *= x;

        1.0 - x * (1.0 - x / 20.0) / 6.0
    }
}