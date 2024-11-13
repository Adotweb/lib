use type_lib::{stringify_value, Value, ValueType};

use std::collections::HashMap;

use std::thread::{self, JoinHandle};
use std::time::Duration;


use std::sync::{
    mpsc::{self, Receiver, Sender},
    Arc, OnceLock,
};

use wgpu::util::DeviceExt;

static RENDER_THREAD: OnceLock<JoinHandle<()>> = OnceLock::new();
static RENDER_THREAD_SENDER: OnceLock<Sender<(String, Value)>> = OnceLock::new();

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop, EventLoopBuilder},
    platform::wayland::EventLoopBuilderExtWayland,
    window::{Window, WindowAttributes, WindowId},
};

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex{
    position : [f32;3],
    color : [f32; 3]
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                }
            ]
        }
    }
}

const VERTICES: &[Vertex] = &[
];

const INDICES: &[u16] = &[];
 


struct State<'a> {
    instance: wgpu::Instance,
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    
    vertex_data : Vec<Vertex>,
    index_data : Vec<u16>,

    vertex_buffer : wgpu::Buffer,
    index_buffer : wgpu::Buffer,
    num_indices : u32
}

impl<'a> State<'a> {
    pub async fn new(window: Arc<Window>) -> State<'a> {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some(""),
                    memory_hints: Default::default(),
                    required_limits: wgpu::Limits::default(),
                    required_features: wgpu::Features::empty(),
                },
                None,
            )
            .await
            .unwrap();

        let size = window.inner_size();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
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
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"), // 1.
                buffers: &[Vertex::desc()],                 // 2.
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                // 3.
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    // 4.
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                count: 1,                         // 2.
                mask: !0,                         // 3.
                alpha_to_coverage_enabled: false, // 4.
            },
            multiview: None, // 5.
            cache: None,     // 6.
        });

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        }
        );
        
        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX,
            }
        );
        let num_indices = INDICES.len() as u32;
        

        Self {
            instance,
            surface,
            device,
            queue,
            config,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            vertex_data : VERTICES.to_vec(),
            index_data : INDICES.to_vec()
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn add_vertices(&mut self, vertices : &[Vertex]){

        let mut all_vertices = self.vertex_data.clone();
        all_vertices.extend_from_slice(vertices); 

        let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label : Some("new vertices"),
            contents : bytemuck::cast_slice(&all_vertices),
            usage : wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST
        });

        let indices = [self.num_indices as u16, self.num_indices as u16 + 1, self.num_indices as u16 + 2];

        let mut all_indices = self.index_data.clone();
        all_indices.extend_from_slice(&indices);

        let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&all_indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        self.vertex_buffer = vertex_buffer;
        self.index_buffer = index_buffer;

        self.vertex_data = all_vertices;
        self.index_data = all_indices;
        println!("{:?} {:?}", self.vertex_data, self.index_data);
    }

    pub fn render(&self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;


        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

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
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline); // 2.  
                                                            
            if self.index_data.len() >= 1{
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16); // 1.
                render_pass.draw_indexed(0..self.index_data.len() as u32, 0, 0..1); // 2. 
            }
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

#[derive(Default)]
struct App<'a> {
    window: Option<Arc<Window>>,
    state: Option<State<'a>>,
    receiver: Option<Receiver<(String, Value)>>,
}

impl<'a> App<'a> {
    fn from_receiver(rec: Receiver<(String, Value)>) -> Self {
        App {
            receiver: Some(rec),
            ..Default::default()
        }
    }
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(WindowAttributes::default())
                .unwrap(),
        );

        self.window = Some(window.clone());

        let state = pollster::block_on(State::new(window.clone()));
        self.state = Some(state);
    }
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: winit::event::WindowEvent,
    ) {
        if let Some(receiver) = &self.receiver {
            if let Ok(rec) = receiver.recv_timeout(Duration::from_millis(16)) {
                match rec.0.as_str(){
                    "message" => println!("{}", stringify_value(rec.1)),
                    "triangle" => {
                        println!("hello");
                        if let Some(state) = self.state.as_mut(){
                            
                            let arr = rec.1.to_arr().unwrap();
                            let vert : Vec<Vertex> = arr.iter().map(|x|{
                                let p: [f32; 3] = x.to_arr().unwrap()
                                    .iter().map(|x|{
                                        x.to_f64().unwrap() as f32
                                    }).collect::<Vec<f32>>()
                                    .try_into().unwrap();
                               
                                Vertex{
                                    position : p,
                                    color : [0.0, 0.0, 0.0]
                                }

                            }).collect();


                            
                            state.add_vertices(&vert); 
                            

                        }
                    },
                    _ => ()
                } 
            }
        }

        match event {
            WindowEvent::Resized(size) => {
                if let Some(state) = self.state.as_mut() {
                    state.resize(size)
                }
            }
            WindowEvent::RedrawRequested => {
                self.window.as_ref().unwrap().request_redraw();

                if let surface = &self.state.as_ref().unwrap().surface {
                } else {
                    return;
                }

                match self.state.as_ref().unwrap().render() {
                    Ok(_) => (),
                    _ => (),
                }
            }
            _ => (),
        }
    }
}

#[no_mangle]
pub extern "Rust" fn start_window(values: HashMap<String, Value>) -> Value {
    let (tx, rx) = mpsc::channel::<(String, Value)>();

    RENDER_THREAD_SENDER.get_or_init(|| tx.clone());

    RENDER_THREAD.get_or_init(|| {
        thread::spawn(move || {
            let mut app = App::from_receiver(rx);

            let event_loop = EventLoop::builder().with_any_thread(true).build().unwrap();
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

            let _ = event_loop.run_app(&mut app);
        })
    });

    Value::nil()
}

#[no_mangle]
pub extern "Rust" fn send_message(values: HashMap<String, Value>) -> Value {
    if let Some(sender) = RENDER_THREAD_SENDER.get() {
        let _ = sender.send(
            ("message".to_string(), values.get("message").unwrap().clone())
        ).unwrap();

        return Value::nil();
    }

    return Value::nil();
}

#[no_mangle]
pub extern "Rust" fn send_triangle(values: HashMap<String, Value>) -> Value{
    if let Some(sender) = RENDER_THREAD_SENDER.get(){
        let _ = sender.send(
            ("triangle".to_string(), values.get("triangle").unwrap().clone())
        );
    }
    return Value::nil()
}

#[no_mangle]
pub extern "Rust" fn value_map() -> HashMap<String, Value> {
    let mut map = HashMap::new();

    map.insert("value".to_string(), Value::number(42.0));

    map.insert(
        "start_window".to_string(),
        Value::lib_function("start_window", vec![], None, None),
    );
    map.insert(
        "send_message".to_string(),
        Value::lib_function("send_message", vec!["message".to_string()], None, None),
    );
    map.insert(
        "send_triangle".to_string(),
        Value::lib_function("send_triangle", vec!["triangle".to_string()], None, None),
    );

    map
}
