use crate::render::RenderSystem;
use anyhow::anyhow;
use ash::prelude::VkResult;
use ash::{khr, vk};
use glfw::{ClientApiHint, Glfw, GlfwReceiver, PWindow, WindowEvent, WindowHint};
use log::debug;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

fn make_window_dark_mode(window: &mut PWindow) {
    #[cfg(windows)]
    {
        use std::ffi::c_void;
        use windows::UI::ViewManagement::{UIColorType, UISettings};
        use windows::Win32::Foundation::HWND;
        use windows::Win32::Graphics::Dwm::{DWMWINDOWATTRIBUTE, DwmSetWindowAttribute};

        let hwnd = HWND(window.get_win32_window());

        let Ok(ui_settings) = UISettings::new() else {
            return;
        };
        let Ok(foreground) = ui_settings.GetColorValue(UIColorType::Foreground) else {
            return;
        };
        let is_dark_mode =
            ((5 * foreground.G as u32) + (2 * foreground.R as u32) + foreground.B as u32)
                > (8 * 128); // (((5 * clr.G) + (2 * clr.R) + clr.B) > (8 * 128));

        if is_dark_mode {
            let is_dark_mode_v = windows::core::BOOL::from(is_dark_mode);
            _ = unsafe {
                DwmSetWindowAttribute(
                    hwnd,
                    DWMWINDOWATTRIBUTE(20),
                    &is_dark_mode_v as *const windows::core::BOOL as *const c_void,
                    size_of::<windows::core::BOOL>() as u32,
                )
            };
        }
    }
}

pub struct WindowSystem {
    glfw: Glfw,
}

pub struct SurfaceInfo {
    pub surface: vk::SurfaceKHR,
    pub surface_fn: khr::surface::Instance,
}

pub struct Window {
    window: PWindow,
    receiver: GlfwReceiver<(f64, WindowEvent)>,
    surface: Option<SurfaceInfo>,
}

impl WindowSystem {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            glfw: glfw::init(glfw::log_errors)?,
        })
    }

    pub fn create_window(
        &mut self,
        size: (u32, u32),
        title: &str,
        window_mode: glfw::WindowMode<'_>,
    ) -> anyhow::Result<Window> {
        self.glfw.default_window_hints();
        self.glfw.window_hint(WindowHint::Resizable(false));
        self.glfw
            .window_hint(WindowHint::ClientApi(ClientApiHint::NoApi));
        let (mut window, receiver) = self
            .glfw
            .create_window(size.0, size.1, title, window_mode)
            .map_or(Err(anyhow!("Failed to create window")), |w| Ok(w))?;

        make_window_dark_mode(&mut window);

        Ok(Window {
            window,
            receiver,
            surface: None,
        })
    }
}

impl Window {
    pub fn poll_events(&mut self, mut f: impl FnMut(f64, WindowEvent, &mut Self)) {
        self.window.glfw.poll_events();
        // collect events into a vec because then we can pass self to the event callback
        for (timestamp, event) in glfw::flush_messages(&self.receiver).collect::<Vec<_>>() {
            f(timestamp, event, self);
        }
    }

    pub fn create_surface(&mut self, render_system: &RenderSystem) -> anyhow::Result<()> {
        let mut surface = MaybeUninit::<vk::SurfaceKHR>::uninit();
        let surface = unsafe {
            self.window
                .create_window_surface(
                    render_system.instance().handle(),
                    std::ptr::null(),
                    surface.as_mut_ptr(),
                )
                .assume_init_on_success(surface)
        }?;
        debug!("Created window surface");

        let surface_fn = khr::surface::Instance::new(render_system.entry(), render_system.instance());

        self.surface = Some(SurfaceInfo{surface, surface_fn});

        Ok(())
    }

    pub fn surface(&self) -> Option<&SurfaceInfo> {
        self.surface.as_ref()
    }
}

impl Deref for Window {
    type Target = PWindow;

    fn deref(&self) -> &Self::Target {
        &self.window
    }
}

impl DerefMut for Window {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.window
    }
}

impl Deref for WindowSystem {
    type Target = Glfw;

    fn deref(&self) -> &Self::Target {
        &self.glfw
    }
}

impl DerefMut for WindowSystem {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.glfw
    }
}

impl Drop for SurfaceInfo {
    fn drop(&mut self) {
        unsafe {
            self.surface_fn.destroy_surface(self.surface, None)
        }
    }
}
