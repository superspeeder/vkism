#![feature(array_try_from_fn)]
#![feature(slice_as_array)]
#![feature(iterator_try_collect)]

pub mod render;
pub mod window;

use crate::render::RenderSystem;
use crate::render::primary_renderer::PrimaryRenderer;
use crate::window::WindowSystem;
use ash::vk::{ClearColorValue, ClearValue};
use glfw::{Action, Key, WindowEvent, WindowMode};
use log::debug;
use render::render_target::SwapchainRenderTarget;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let mut window_system = WindowSystem::new()?;
    let render_system = RenderSystem::new(&window_system)?;
    let mut window = window_system.create_window((800, 600), "Hello!", WindowMode::Windowed)?;
    window.create_surface(&render_system)?;

    let mut primary_renderer = PrimaryRenderer::new(&render_system)?;
    let mut window_target = SwapchainRenderTarget::new(window, &render_system)?;

    primary_renderer.set_clear_value(
        0,
        Some(ClearValue {
            color: ClearColorValue {
                float32: [0.25, 0.5, 0.5, 1.0],
            },
        }),
    );

    window_target.set_key_polling(true);

    let mut this_frame = window_system.get_time();
    let mut last_frame = this_frame - 1.0 / 60.0;
    let mut delta_time = this_frame - last_frame;

    let mut frame_counter: usize = 0;

    while !window_target.should_close() {
        window_target.poll_events(|_, event, window| match event {
            WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                window.set_should_close(true);
            }
            _ => (),
        });

        primary_renderer.render_to_target(&mut window_target, |_cmd, _frame_info| {});


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
