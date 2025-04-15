#![feature(array_try_from_fn)]
#![feature(slice_as_array)]

mod window;
mod render;

use glfw::{Action, Key, WindowEvent, WindowMode};
use crate::render::primary_renderer::PrimaryRenderer;
use crate::render::RenderSystem;
use crate::window::WindowSystem;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let mut window_system = WindowSystem::new()?;
    let mut render_system = RenderSystem::new(&window_system)?;
    let mut window = window_system.create_window((800, 600), "Hello!", WindowMode::Windowed)?;
    window.create_surface(&render_system)?;

    // let mut render_target =
    let primary_renderer = PrimaryRenderer::new(&render_system)?;


    window.set_key_polling(true);

    while !window.should_close() {
        window.poll_events(|_, event, window| {
            match event {
                WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                    window.set_should_close(true);
                }
                _ => ()
            }
        });
    }

    Ok(())
}
