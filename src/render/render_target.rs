use std::ops::{Deref, DerefMut};

use crate::render::RenderSystem;
use crate::window::Window;
use anyhow::anyhow;
use ash::{
    khr,
    vk::{
        self, AccessFlags2, ColorSpaceKHR, ComponentMapping, ComponentSwizzle,
        CompositeAlphaFlagsKHR, Extent2D, Format, ImageAspectFlags, ImageLayout,
        ImageSubresourceRange, ImageUsageFlags, ImageViewCreateInfo, ImageViewType, PresentInfoKHR,
        PresentModeKHR, SharingMode,
    },
};

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
    fn render_frame_prelude(
        &mut self,
        sync_info: &FrameSyncInfo,
    ) -> anyhow::Result<FrameRenderInfo>;
    fn render_frame_postlude(
        &mut self,
        sync_info: &FrameSyncInfo,
        frame_render_info: FrameRenderInfo,
    );
}

pub trait RenderTargetExt: RenderTarget {
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
        let Ok(frame_render_info) = self.render_frame_prelude(sync_info) else {
            return;
        };
        f(&frame_render_info);
        self.render_frame_postlude(sync_info, frame_render_info);
    }
}

pub struct SwapchainRenderTarget {
    window: Window, // takes ownership of the window since we have to ensure that the window lasts long enough
    swapchain: vk::SwapchainKHR,
    device: ash::Device,

    extent: vk::Extent2D,
    format: vk::SurfaceFormatKHR,

    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,

    swapchain_fn: khr::swapchain::Device,
    present_queue: vk::Queue,
}

impl FrameSyncInfo {
    pub fn new(render_system: &RenderSystem) -> anyhow::Result<Self> {
        Ok(Self {
            image_available: render_system.create_semaphore()?,
            render_finished: render_system.create_semaphore()?,
        })
    }
}

impl SwapchainRenderTarget {
    pub fn new(
        window: Window,
        render_system: &RenderSystem,
    ) -> anyhow::Result<SwapchainRenderTarget> {
        let surface = window
            .surface()
            .ok_or(anyhow!("Window surface not created"))?;

        let capabilities = unsafe {
            surface.surface_fn.get_physical_device_surface_capabilities(
                render_system.physical_device(),
                surface.surface,
            )
        }?;

        let extent = if capabilities.current_extent.height == u32::MAX {
            let window_extent = window.get_framebuffer_size();
            Extent2D {
                width: u32::clamp(
                    window_extent.0 as u32,
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: u32::clamp(
                    window_extent.1 as u32,
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        } else {
            capabilities.current_extent
        };

        let image_count = if capabilities.max_image_count > 0 {
            u32::min(
                capabilities.max_image_count,
                capabilities.min_image_count + 1,
            )
        } else {
            capabilities.min_image_count + 1
        };

        let formats = unsafe {
            surface.surface_fn.get_physical_device_surface_formats(
                render_system.physical_device(),
                surface.surface,
            )
        }?;

        let format = formats
            .iter()
            .find(|f| {
                f.color_space == ColorSpaceKHR::SRGB_NONLINEAR && f.format == Format::B8G8R8A8_SRGB
            })
            .cloned()
            .or(formats.first().cloned())
            .ok_or(anyhow!("Invalid surface"))?;

        let present_modes = unsafe {
            surface
                .surface_fn
                .get_physical_device_surface_present_modes(
                    render_system.physical_device(),
                    surface.surface,
                )
        }?;

        let mut present_mode = PresentModeKHR::FIFO;
        for mode in present_modes {
            if mode == PresentModeKHR::MAILBOX {
                present_mode = mode;
                break;
            }
        }

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .clipped(true)
            .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
            .image_array_layers(1)
            .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(SharingMode::EXCLUSIVE)
            .surface(surface.surface)
            .pre_transform(capabilities.current_transform)
            .image_extent(extent)
            .image_color_space(format.color_space)
            .image_format(format.format)
            .min_image_count(image_count)
            .present_mode(present_mode);

        let swapchain_fn =
            khr::swapchain::Device::new(render_system.instance(), render_system.device());

        let swapchain = unsafe { swapchain_fn.create_swapchain(&swapchain_create_info, None) }?;

        let images = unsafe { swapchain_fn.get_swapchain_images(swapchain) }?;

        let image_views = images
            .iter()
            .cloned()
            .map(|i| {
                let create_info = ImageViewCreateInfo::default()
                    .image(i)
                    .components(
                        ComponentMapping::default()
                            .r(ComponentSwizzle::R)
                            .g(ComponentSwizzle::G)
                            .b(ComponentSwizzle::B)
                            .a(ComponentSwizzle::A),
                    )
                    .format(format.format)
                    .subresource_range(ImageSubresourceRange {
                        aspect_mask: ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .view_type(ImageViewType::TYPE_2D);

                unsafe { render_system.device().create_image_view(&create_info, None) }
            })
            .try_collect::<Vec<_>>()?;

        Ok(Self {
            window,
            swapchain,
            device: render_system.device().clone(),
            extent,
            format,
            images,
            image_views,
            swapchain_fn,
            present_queue: render_system.queues().present,
        })
    }
}

impl RenderTarget for SwapchainRenderTarget {
    fn render_frame_prelude(
        &mut self,
        sync_info: &FrameSyncInfo,
    ) -> anyhow::Result<FrameRenderInfo> {
        let (image_index, _) = unsafe {
            self.swapchain_fn.acquire_next_image(
                self.swapchain,
                u64::MAX,
                sync_info.image_available,
                vk::Fence::null(),
            )
        }?;
        let image = self.images[image_index as usize];
        let image_view = self.image_views[image_index as usize];

        Ok(FrameRenderInfo {
            color_attachments: vec![FrameRenderAttachment {
                image,
                image_view,
                format: self.format.format,
                initial_state: FrameRenderAttachmentImageStateExternal {
                    layout: ImageLayout::UNDEFINED,
                    access: AccessFlags2::NONE,
                    queue_family: 0,
                },
                final_state: FrameRenderAttachmentImageStateExternal {
                    layout: ImageLayout::PRESENT_SRC_KHR,
                    access: AccessFlags2::NONE,
                    queue_family: 0,
                },
            }],
            extent: self.extent,
            image_index,
        })
    }

    fn render_frame_postlude(
        &mut self,
        sync_info: &FrameSyncInfo,
        frame_render_info: FrameRenderInfo,
    ) {
        unsafe {
            let _ = self.swapchain_fn.queue_present(
                self.present_queue,
                &PresentInfoKHR::default()
                    .image_indices(&[frame_render_info.image_index])
                    .wait_semaphores(&[sync_info.render_finished])
                    .swapchains(&[self.swapchain]),
            );
        };
    }
}

impl Deref for SwapchainRenderTarget {
    type Target = Window;

    fn deref(&self) -> &Self::Target {
        &self.window
    }
}

impl DerefMut for SwapchainRenderTarget {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.window
    }
}

impl Drop for SwapchainRenderTarget {
    fn drop(&mut self) {
        unsafe {
            for iv in self.image_views.iter().cloned() {
                self.device.destroy_image_view(iv, None);
            }

            self.swapchain_fn.destroy_swapchain(self.swapchain, None);
        }
    }
}
