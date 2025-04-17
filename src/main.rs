#![feature(array_try_from_fn)]
#![feature(slice_as_array)]
#![feature(iterator_try_collect)]

pub mod render;
pub mod window;

use crate::render::RenderSystem;
use crate::render::command_buffer::RenderingRecorder;
use crate::render::pipeline::{
    ColorBlendingDescription, GraphicsPipeline, GraphicsPipelineDescription,
    MultisamplingDescription, PipelineLayout, PipelineLayoutDescription,
    PipelineRenderCompatibility, RasterizerDescription, VertexLayout, standard_blend_attachment,
    standard_vertex_fragment_stages, standard_viewport_scissor_from_extent,
};
use crate::render::primary_renderer::PrimaryRenderer;
use crate::render::shader::ShaderModule;
use crate::window::WindowSystem;
use ash::vk;
use ash::vk::{
    BlendFactor, BlendOp, ClearColorValue, ClearValue, ColorComponentFlags, CullModeFlags, Format,
    FrontFace, Offset2D, PolygonMode, PrimitiveTopology, Rect2D, SampleCountFlags,
    ShaderStageFlags, Viewport,
};
use glfw::{Action, Key, WindowEvent, WindowMode};
use log::debug;
use render::render_target::SwapchainRenderTarget;
use shaderc::ShaderKind;
use std::ops::Not;
use std::rc::Rc;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let mut window_system = WindowSystem::new()?;
    let render_system = RenderSystem::new(&window_system)?;
    let mut window = window_system.create_window((800, 600), "Hello!", WindowMode::Windowed)?;
    window.create_surface(&render_system)?;

    let mut primary_renderer = PrimaryRenderer::new(&render_system)?;
    let mut window_target = SwapchainRenderTarget::new(window, &render_system)?;

    primary_renderer.set_clear_color(0, Some([0.25, 0.5, 0.5, 1.0]));

    window_target.set_key_polling(true);

    let mut this_frame = window_system.get_time();
    let mut last_frame = this_frame - 1.0 / 60.0;
    let mut delta_time = this_frame - last_frame;

    let mut frame_counter: usize = 0;

    let vertex_shader = Rc::new(ShaderModule::load_glsl(
        &render_system,
        "res/main.vert",
        ShaderKind::Vertex,
    )?);

    let fragment_shader = Rc::new(ShaderModule::load_glsl(
        &render_system,
        "res/main.frag",
        ShaderKind::Fragment,
    )?);

    let pipeline_layout = Rc::new(PipelineLayout::new(
        &render_system,
        &PipelineLayoutDescription {
            push_constant_ranges: vec![],
            descriptor_set_layouts: vec![],
        },
    )?);

    let extent = window_target.get_swapchain_extent();

    let pipeline = GraphicsPipeline::new(
        &render_system,
        &GraphicsPipelineDescription {
            layout: pipeline_layout,
            rendering_compatibility: PipelineRenderCompatibility::simple_from_format(
                window_target.get_swapchain_format(),
            ),
            shader_stages: standard_vertex_fragment_stages(
                vertex_shader.clone(),
                fragment_shader.clone(),
            ),
            vertex_layout: VertexLayout::default(),
            primitive_topology: PrimitiveTopology::TRIANGLE_LIST,
            allow_primitive_restart: false,
            tessellator_patch_control_points: 0,
            viewports: vec![standard_viewport_scissor_from_extent(extent)],
            rasterizer: RasterizerDescription::default(),
            multisampling: MultisamplingDescription::default(),
            depth_stencil: None,
            color_blending: ColorBlendingDescription {
                attachments: vec![standard_blend_attachment()],
                ..ColorBlendingDescription::default()
            },
            dynamic_states: vec![],
        },
    )?;

    while !window_target.should_close() {
        window_target.poll_events(|_, event, window| match event {
            WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                window.set_should_close(true);
            }
            _ => (),
        });

        primary_renderer.render_to_target(&mut window_target, |cmd, _frame_info| {
            cmd.bind_graphics_pipeline(&pipeline);
            cmd.draw(3, 1, 0, 0);
        });

        frame_counter += 1;
        if frame_counter % 1000 == 0 {
            debug!("FPS: {:?}", 1.0 / delta_time);
        }

        last_frame = this_frame;
        this_frame = window_system.get_time();
        delta_time = this_frame - last_frame;
    }

    unsafe { _ = render_system.device().device_wait_idle() };

    Ok(())
}
