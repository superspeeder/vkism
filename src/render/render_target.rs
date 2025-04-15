use crate::render::RenderSystem;
use crate::window::Window;
use ash::vk;

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct FrameRenderAttachmentImageStateExternal {
    pub layout: vk::ImageLayout,
    pub access: vk::AccessFlags2,
    pub queue_family: u32,
}

pub struct FrameRenderAttachment {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub format: vk::Format,
    pub initial_state: FrameRenderAttachmentImageStateExternal,
    pub final_state: FrameRenderAttachmentImageStateExternal,
}

pub struct FrameRenderInfo {
    pub color_attachments: Vec<FrameRenderAttachment>,
    pub extent: vk::Extent2D,
    pub image_index: u32,
}

pub struct FrameSyncInfo {
    pub image_available: vk::Semaphore,
    pub render_finished: vk::Semaphore,
}

pub trait RenderTarget {
    fn render_frame_prelude(&mut self, sync_info: &FrameSyncInfo) -> FrameRenderInfo;
    fn render_frame_postlude(&mut self, sync_info: &FrameSyncInfo, frame_render_info: FrameRenderInfo);
}

pub trait RenderTargetExt {
    ///
    /// Call this to render a frame to this target.
    /// Due to the fact that most render targets will require external synchronization, you have to pass in synchronization objects.
    /// Maintaining synchronization of other resources is outside the requirements of this function (i.e. if you use a frames-in-flight model and have a variety of things per frame, it is up to you to ensure that you don't use resources which are already in use.
    /// The passed in sync info is specifically used to signal availability of the render target.
    ///
    /// >
    /// > WARNING: this is a super low level function, it is likely preferred to use a more high level interface for most applications, which will generally manage most synchronization and resource issues for you.
    /// >
    ///
    fn render_frame(&mut self, sync_info: &FrameSyncInfo, f: impl FnOnce(&FrameRenderInfo));
}

impl<T: RenderTarget + ?Sized> RenderTargetExt for T {
    fn render_frame(&mut self, sync_info: &FrameSyncInfo, f: impl FnOnce(&FrameRenderInfo)) {
        let frame_render_info = self.render_frame_prelude(sync_info);
        f(&frame_render_info);
        self.render_frame_postlude(sync_info, frame_render_info);
    }
}

pub struct SwapchainRenderTarget {
    window: Window, // takes ownership of the window since we have to ensure that the window lasts long enough
    swapchain: vk::SwapchainKHR,
}

impl FrameSyncInfo {
    pub fn new(render_system: &RenderSystem) -> anyhow::Result<Self> {
        Ok(Self {
            image_available: render_system.create_semaphore()?,
            render_finished: render_system.create_semaphore()?,
        })
    }
}
