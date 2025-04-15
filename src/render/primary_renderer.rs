use crate::render::RenderSystem;
use crate::render::command_buffer::{CommandBuffer, CommandRecorder, ImageTransition};
use crate::render::render_target::{FrameRenderAttachmentImageStateExternal, FrameRenderInfo, FrameSyncInfo, RenderTarget, RenderTargetExt};
use ash::vk;
use ash::vk::{AccessFlags, AccessFlags2, AttachmentLoadOp, AttachmentStoreOp, ClearValue, CommandBufferBeginInfo, CommandBufferUsageFlags, ImageAspectFlags, ImageLayout, ImageSubresourceRange, PipelineStageFlags2, Rect2D, RenderingAttachmentInfo, RenderingInfo};

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub type FrameSet<T> = [T; MAX_FRAMES_IN_FLIGHT];

pub struct PrimaryRenderer {
    device: ash::Device,
    frame_sync_infos: FrameSet<FrameSyncInfo>,
    fences: FrameSet<vk::Fence>,
    command_buffers: FrameSet<CommandBuffer>,

    render_area: Option<vk::Rect2D>,
    clear_values: Vec<Option<vk::ClearValue>>,

    current_frame: usize,
}

impl PrimaryRenderer {
    pub fn new(render_system: &RenderSystem) -> anyhow::Result<PrimaryRenderer> {
        let frame_sync_infos = std::array::try_from_fn(|_| FrameSyncInfo::new(render_system))?;
        let fences = std::array::try_from_fn(|_| render_system.create_fence(true))?;
        let command_buffers = render_system
            .create_command_buffers::<MAX_FRAMES_IN_FLIGHT>(vk::CommandBufferLevel::PRIMARY)?;

        Ok(Self {
            device: render_system.device().clone(),
            frame_sync_infos,
            fences,
            command_buffers,
            render_area: None,
            clear_values: Vec::new(),
            current_frame: 0,
        })
    }

    pub fn begin_frame(&mut self) {
        let fence = self.fences[self.current_frame];
        unsafe {
            let _ = self.device.wait_for_fences(&[fence], true, u64::MAX);
        }
    }

    pub fn end_frame(&mut self) {
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    pub fn set_clear_value(&mut self, index: usize, value: Option<vk::ClearValue>) {
        if self.clear_values.len() <= index {
            self.clear_values.resize(index, None);
        }

        self.clear_values[index] = value;
    }

    pub fn set_render_area(&mut self, area: Option<vk::Rect2D>) {
        self.render_area = area;
    }

    pub fn render_to_target(
        &mut self,
        target: &mut dyn RenderTarget,
        f: impl FnOnce(&CommandRecorder, &FrameRenderInfo),
    ) {
        self.begin_frame();

        RenderTargetExt::render_frame(target, &self.frame_sync_infos[self.current_frame], |render_info| {
            let Ok(cmd) = self.command_buffers[self.current_frame].begin(Some(
                CommandBufferBeginInfo::default().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )) else {
                return;
            };

            let mut image_transitions_1 = render_info
                .color_attachments
                .iter()
                .filter_map(|attachment| {
                    if attachment.initial_state.layout == ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                        && attachment.initial_state.access == AccessFlags2::COLOR_ATTACHMENT_WRITE
                        && attachment.initial_state.queue_family == 0
                    {
                        return None;
                    }
                    Some(ImageTransition {
                        image: attachment.image,
                        subresource_range: ImageSubresourceRange {
                            aspect_mask: ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        src_state: (
                            PipelineStageFlags2::TOP_OF_PIPE,
                            attachment.initial_state.layout,
                            attachment.initial_state.access,
                            attachment.initial_state.queue_family,
                        ),
                        dst_state: (
                            PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                            ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            AccessFlags2::COLOR_ATTACHMENT_WRITE,
                            0,
                        ),
                    })
                })
                .collect::<Vec<_>>();

            let mut image_transitions_2 = render_info
                .color_attachments
                .iter()
                .filter_map(|attachment| {
                    if attachment.final_state.layout == ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                        && attachment.final_state.access == AccessFlags2::COLOR_ATTACHMENT_WRITE
                        && attachment.final_state.queue_family == 0
                    {
                        return None;
                    }

                    Some(ImageTransition {
                        image: attachment.image,
                        subresource_range: ImageSubresourceRange {
                            aspect_mask: ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        src_state: (
                            PipelineStageFlags2::TOP_OF_PIPE,
                            ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            AccessFlags2::COLOR_ATTACHMENT_WRITE,
                            0,
                        ),
                        dst_state: (
                            PipelineStageFlags2::BOTTOM_OF_PIPE,
                            attachment.final_state.layout,
                            attachment.final_state.access,
                            attachment.final_state.queue_family,
                        ),
                    })
                })
                .collect::<Vec<_>>();

            if !image_transitions_1.is_empty() {
                cmd.image_transitions(image_transitions_1.as_slice())
            }

            {
                let color_attachments = render_info.color_attachments.iter()
                    .enumerate()
                    .map(|(i, attachment)| {
                        let clear_value = self.clear_values.get(i).cloned().unwrap_or(None);

                        RenderingAttachmentInfo::default()
                            .image_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(match clear_value { None => AttachmentLoadOp::LOAD, Some(_) => AttachmentLoadOp::CLEAR })
                            .clear_value(clear_value.unwrap_or(ClearValue::default()))
                            .store_op(AttachmentStoreOp::STORE)
                            .image_view(attachment.image_view)
                    }).collect::<Vec<_>>();

                let rendering_info = RenderingInfo::default()
                    .render_area(self.render_area.unwrap_or(Rect2D::from(render_info.extent)))
                    .color_attachments(color_attachments.as_slice())
                    .layer_count(1)
                    .view_mask(0);

                let cmd = cmd.begin_rendering(rendering_info);
            }

            if !image_transitions_2.is_empty() {
                cmd.image_transitions(image_transitions_2.as_slice())
            }
        });

        self.end_frame();
    }
}
