extern crate byte_slice_cast;

#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch="wasm32")]
extern crate console_error_panic_hook;

use std::iter;

use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};
use winit::keyboard::PhysicalKey;
use winit::keyboard::KeyCode;
use winit::window::Fullscreen;
use winit::dpi::PhysicalPosition;
mod texture;
mod vertex;
mod time;
mod math;
use fitsrs::card::Value;
use fitsrs::HDU;

use time::Clock;
use vertex::Vertex;
use texture::Texture;
use crate::math::Vec4;

struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    window: &'a Window,

    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    //num_indices: u32,

    //cube: texture::Texture,
    //texture_bind_group_layout: wgpu::BindGroupLayout,
    diffuse_bind_group: wgpu::BindGroup,

    // uniforms
    rot_mat_buf: wgpu::Buffer,
    window_size_buf: wgpu::Buffer,
    time_buf: wgpu::Buffer,
    cam_origin_buf: wgpu::Buffer,
    cuts_buf: wgpu::Buffer,

    clock: Clock,
}

use fitsrs::Fits;
use memmap2::Mmap;
use std::fs::File;
use std::io::Cursor;

use crate::math::Mat4;
impl<'a> State<'a> {
    async fn new(window: &'a Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch="wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch="wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    // favor performane over the memory usage
                    memory_hints: Default::default(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web, we'll have to disable some.
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits {
                            max_texture_dimension_3d: 512,
                            ..wgpu::Limits::downlevel_webgl2_defaults()
                        }
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![surface_format.add_srgb_suffix()],
            desired_maximum_frame_latency: 2,
        };

        /*let bytes = include_bytes!("../img/map.png");
        let img = image::load_from_memory(bytes).unwrap();
        let map_texture = texture::Texture::from_image(&device, &queue, &img, "map.png");*/

        let file = File::open("cubes/NGC3198_cube.fits").unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };

        let mut hdu_list = Fits::from_reader(Cursor::new(mmap));
        let hdu = hdu_list.next().unwrap().unwrap();

        let cube = match hdu {
            HDU::Primary(hdu) => {
                let header = hdu.get_header();

                if let (Some(Value::Integer {value: w, .. }), Some(Value::Integer {value: h, .. }), Some(Value::Integer {value: d, .. })) = (header.get("NAXIS1"), header.get("NAXIS2"), header.get("NAXIS3")) {
                    let image = hdu_list.get_data(&hdu);

                    Texture::from_raw_bytes::<f32>(
                        &device,
                        &queue,
                        Some(image.raw_bytes()),
                        (*w as u32, *h as u32, *d as u32),
                        4,
                        "cube"
                    )
                } else {
                    unreachable!()
                }
            }
            _ => unreachable!()
        };

        // Uniform buffer
        let rot_mat_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rot matrix uniform"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let time_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("time in secs since starting"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cam_origin_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cam origin"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cuts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cuts"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let window_size_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("window size uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D3,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    // rot matrix uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<Mat4<f32>>() as _,
                            ),
                        },
                        count: None,
                    },
                    // window size uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<Vec4<f32>>() as wgpu::BufferAddress,
                            ),
                        },
                        count: None,
                    },
                    // time uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<Vec4<f32>>() as wgpu::BufferAddress,
                            ),
                        },
                        count: None,
                    },
                    // cam origin uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<Vec4<f32>>() as wgpu::BufferAddress,
                            ),
                        },
                        count: None,
                    },
                    // cuts uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<Vec4<f32>>() as wgpu::BufferAddress,
                            ),
                        },
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&cube.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&cube.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &rot_mat_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(
                            std::mem::size_of::<Mat4<f32>>() as wgpu::BufferAddress
                        ),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &window_size_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(
                            16
                        ),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &time_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(
                            16
                        ),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &cam_origin_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(
                            16
                        ),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &cuts_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(
                            16
                        ),
                    }),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        // uniform buffer
        let vs_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cube vert shader"),
                source: wgpu::ShaderSource::Glsl {
                    shader: std::str::from_utf8(&std::fs::read("src\\shaders\\cube.vert").unwrap()).unwrap().into(),
                    stage: naga::ShaderStage::Vertex,
                    defines: Default::default()
                }
            });
        let fs_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cube frag shader"),
                source: wgpu::ShaderSource::Glsl {
                    shader: std::str::from_utf8(&std::fs::read("src\\shaders\\cube.frag").unwrap()).unwrap().into(),
                    stage: naga::ShaderStage::Fragment,
                    defines: Default::default()
                },
            });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_shader,
                entry_point: "main",
                compilation_options: Default::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_shader,
                entry_point: "main",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None, // 5.
            cache: None, // 6.
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&[
                Vertex { ndc: [-1.0, -1.0] },
                Vertex { ndc: [1.0, -1.0] },
                Vertex { ndc: [1.0, 1.0] },
                Vertex { ndc: [-1.0, 1.0] }
            ]),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&[0, 1, 2, 0, 2, 3]),
            usage: wgpu::BufferUsages::INDEX,
        });
        //let num_indices = indices.len() as u32;

        // set the initial cut values
        queue.write_buffer(
            &cuts_buf,
            0,
            bytemuck::bytes_of(&[1.0 as f32, 0.0, 0.0, 0.0]),
        );

        let clock = Clock::now();
        Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            vertex_buffer,
            index_buffer,

            //texture_bind_group_layout,
            diffuse_bind_group,

            // uniforms
            window_size_buf,
            rot_mat_buf,
            time_buf,
            cam_origin_buf,
            cuts_buf,

            clock,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            #[cfg(target_arch="wasm32")] {
                new_size.width = new_size.width.min(wgpu::Limits::downlevel_webgl2_defaults().max_texture_dimension_2d);
                new_size.height = new_size.height.min(wgpu::Limits::downlevel_webgl2_defaults().max_texture_dimension_2d);    
            }

            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
        self.queue.write_buffer(
            &self.window_size_buf,
            0,
            bytemuck::bytes_of(&[self.size.width as f32, self.size.height as f32, 0.0, 0.0]),
        );
    }

    #[allow(unused_variables)]
    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {
        let elapsed = self.clock.elapsed_as_secs();

        let rot = Mat4::from_angle_y(cgmath::Rad(elapsed));
        let rot: &[[f32; 4]; 4] = rot.as_ref();

        self.queue
            .write_buffer(&self.rot_mat_buf, 0, bytemuck::bytes_of(rot));
        self.queue
            .write_buffer(&self.time_buf, 0, bytemuck::bytes_of(&[elapsed, 0.0, 0.0, 0.0]));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(())
        }

        if let Ok(frame) = self.surface.get_current_texture() {
            let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.config.format.add_srgb_suffix()),
                ..Default::default()
            });
    
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });
    
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.01,
                                g: 0.01,
                                b: 0.01,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
    
                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..6, 0, 0..1);
            }
    
            self.queue.submit(iter::once(encoder.finish()));
            frame.present();
        }

        Ok(())
    }
}

#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
pub async fn run() {
    #[cfg(target_arch = "wasm32")]
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    #[cfg(target_arch = "wasm32")]
    console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    let builder = WindowBuilder::new();

    #[cfg(target_arch = "wasm32")]
    {   
        use wasm_bindgen::JsCast;
        use winit::platform::web::WindowBuilderExtWebSys;
        let canvas = web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();

        builder = builder.with_canvas(Some(canvas));
    }
    let window = builder.with_title("Astronomical cube visualizer")
        .build(&event_loop).unwrap();

    // Winit prevents sizing with CSS, so we have to set
    // the size manually when on web.
    #[cfg(target_arch = "wasm32")]
    {
        use winit::dpi::LogicalSize;
        let _ = window.request_inner_size(LogicalSize::new(768, 512));
    }

    let mut state = State::new(&window).await;

    let mut panning = false;
    let mut cuts = false;
    let mut cursor_pos = PhysicalPosition::new(0.0, 0.0);
    let mut start_cursor_pos = PhysicalPosition::new(0.0, 0.0);

    // move variable
    let mut delta = 0.0;
    let mut theta = 0.0;
    let mut dtheta = 0.0;
    let mut ddelta: f64 = 0.0;

    // cuts
    let mut scale = 1.0;
    let mut offset = 0.0;
    let mut dscale = 0.0;
    let mut doffset = 0.0;

    event_loop.run(move |event, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window.id() => {
                if !state.input(event) {
                    match event {
                        #[cfg(not(target_arch="wasm32"))]
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            event:
                                KeyEvent {
                                    state: ElementState::Pressed,
                                    physical_key: PhysicalKey::Code(KeyCode::Escape),
                                    ..
                                },
                            ..
                        } => control_flow.exit(),
                        WindowEvent::KeyboardInput {
                            event:
                                KeyEvent {
                                    state: ElementState::Pressed,
                                    physical_key: PhysicalKey::Code(KeyCode::Enter),
                                    ..
                                },
                            ..
                        } => {
                            // toggle fullscreen
                            state.window.set_fullscreen(Some(Fullscreen::Borderless(None)));
                        },
                        WindowEvent::Resized(physical_size) => state.resize(*physical_size),
                        WindowEvent::RedrawRequested => {
                            state.update();
                            match state.render() {
                                Ok(_) => {}
                                // Reconfigure the surface if lost
                                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                                // The system is out of memory, we should probably quit
                                Err(wgpu::SurfaceError::OutOfMemory) => control_flow.exit(),
                                // All other errors (Outdated, Timeout) should be resolved by the next frame
                                Err(e) => { eprintln!("{}", e); },
                            }
                        }
                        // Moving
                        WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                            panning = true;
                            start_cursor_pos = cursor_pos;
                            dtheta = 0.0;
                            ddelta = 0.0;
                        }
                        WindowEvent::MouseInput { state: ElementState::Released, button: MouseButton::Left, .. } => {
                            panning = false;
                            theta += dtheta;
                            delta += ddelta;

                            delta = delta.clamp(-std::f64::consts::PI * 0.5 + 1e-3, std::f64::consts::PI * 0.5 - 1e-3);
                        }
                        // Change cuts
                        WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Right, .. } => {
                            cuts = true;
                            start_cursor_pos = cursor_pos;
                            dscale = 0.0;
                            doffset = 0.0;
                        }
                        WindowEvent::MouseInput { state: ElementState::Released, button: MouseButton::Right, .. } => {
                            cuts = false;
                            scale += dscale;
                            offset += doffset;
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            cursor_pos = *position;

                            if panning {
                                let dx = (cursor_pos.x - start_cursor_pos.x) / ((state.size.width as f64) * 0.5);
                                let dy = (cursor_pos.y - start_cursor_pos.y) / ((state.size.height as f64) * 0.5);

                                dtheta = 2.0*dx;
                                ddelta = dy;

                                let d = (delta as f32 + ddelta as f32).clamp(-std::f32::consts::PI * 0.5 + 1e-3, std::f32::consts::PI * 0.5 - 1e-3);

                                state.queue.write_buffer(
                                    &state.cam_origin_buf,
                                    0,
                                    bytemuck::bytes_of(&[theta as f32 + dtheta as f32, d, 0.0, 0.0]),
                                );
                            } else if cuts {
                                let dx = (cursor_pos.x - start_cursor_pos.x) / ((state.size.width as f64) * 0.5);
                                let dy = (cursor_pos.y - start_cursor_pos.y) / ((state.size.height as f64) * 0.5);

                                dscale = dy;
                                doffset = dx;

                                state.queue.write_buffer(
                                    &state.cuts_buf,
                                    0,
                                    bytemuck::bytes_of(&[scale as f32 + dscale as f32, offset as f32 + doffset as f32, 0.0, 0.0]),
                                );
                            }
                        }
                        _ => {}
                    }
                }
            }
            // ... at the end of the WindowEvent block
            Event::AboutToWait => {
                // RedrawRequested will only trigger once unless we manually
                // request it.
                state.window.request_redraw();
            }
            _ => {}
        }
    }).unwrap();
}
