use crate::render::pipeline::GraphicsPipeline;
use ash::vk;
use ash::vk::{
    CommandBufferBeginInfo, CommandBufferUsageFlags, DependencyInfo, ImageMemoryBarrier2,
    PipelineBindPoint, RenderingInfo, StructureType,
};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use log::info;

pub struct CommandBuffer {
    command_buffer: vk::CommandBuffer,
    device: ash::Device,
}

pub struct CommandRecorder<'a>(&'a mut CommandBuffer);

pub trait GenericCommandRecorder<'a> {
    fn command_recorder(&self) -> &CommandRecorder<'a>;
}

impl CommandBuffer {
    pub fn wrap(device: &ash::Device, command_buffer: vk::CommandBuffer) -> Self {
        Self {
            command_buffer,
            device: device.clone(),
        }
    }

    #[inline]
    pub fn begin(
        &mut self,
        begin_info: Option<CommandBufferBeginInfo>,
    ) -> anyhow::Result<CommandRecorder<'_>> {
        CommandRecorder::begin(self, begin_info)
    }

    pub fn handle(&self) -> vk::CommandBuffer {
        self.command_buffer
    }
}

pub struct ImageTransition {
    pub image: vk::Image,
    pub subresource_range: vk::ImageSubresourceRange,
    pub src_state: (
        vk::PipelineStageFlags2,
        vk::ImageLayout,
        vk::AccessFlags2,
        u32,
    ),
    pub dst_state: (
        vk::PipelineStageFlags2,
        vk::ImageLayout,
        vk::AccessFlags2,
        u32,
    ),
}

impl<'a> CommandRecorder<'a> {
    pub fn begin(
        command_buffer: &'a mut CommandBuffer,
        begin_info: Option<CommandBufferBeginInfo>,
    ) -> anyhow::Result<Self> {
        unsafe {
            let begin_info = begin_info.unwrap_or(CommandBufferBeginInfo::default());

            command_buffer
                .device
                .begin_command_buffer(command_buffer.command_buffer, &begin_info)
        }?;

        Ok(Self(command_buffer))
    }

    pub fn pipeline_barrier(&self, dependency_info: DependencyInfo) {
        unsafe {
            let _ = self
                .0
                .device
                .cmd_pipeline_barrier2(self.0.command_buffer, &dependency_info);
        }
    }

    pub fn image_transitions(&self, transitions: &[ImageTransition]) {
        let image_memory_barriers = transitions
            .iter()
            .map(|t| ImageMemoryBarrier2 {
                s_type: StructureType::IMAGE_MEMORY_BARRIER_2,
                p_next: std::ptr::null_mut(),
                src_stage_mask: t.src_state.0,
                src_access_mask: t.src_state.2,
                dst_stage_mask: t.dst_state.0,
                dst_access_mask: t.dst_state.2,
                old_layout: t.src_state.1,
                new_layout: t.dst_state.1,
                src_queue_family_index: t.src_state.3,
                dst_queue_family_index: t.dst_state.3,
                image: t.image,
                subresource_range: t.subresource_range,
                _marker: PhantomData,
            })
            .collect::<Vec<_>>();

        let dependency_info =
            DependencyInfo::default().image_memory_barriers(image_memory_barriers.as_slice());

        self.pipeline_barrier(dependency_info);
    }

    #[inline]
    pub fn image_transition(&self, transition: ImageTransition) {
        self.image_transitions(&[transition]);
    }

    #[inline]
    pub fn begin_rendering(
        &self,
        rendering_info: RenderingInfo,
    ) -> DynamicRenderingRecorder<'_, 'a> {
        DynamicRenderingRecorder::begin(self, rendering_info)
    }
}

impl<'a> GenericCommandRecorder<'a> for CommandRecorder<'a> {
    fn command_recorder(&self) -> &CommandRecorder<'a> {
        self
    }
}

impl Drop for CommandRecorder<'_> {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.end_command_buffer(self.command_buffer);
        }
    }
}

impl Deref for CommandRecorder<'_> {
    type Target = CommandBuffer;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

pub trait RenderingRecorder<'a>: GenericCommandRecorder<'a> {
    fn bind_graphics_pipeline(&self, pipeline: &GraphicsPipeline) {
        unsafe {
            let cmd = self.command_recorder();
            cmd.device.cmd_bind_pipeline(
                cmd.command_buffer,
                PipelineBindPoint::GRAPHICS,
                pipeline.handle(),
            );
        }
    }

    fn draw(&self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
        unsafe {
            let cmd = self.command_recorder();
            cmd.device.cmd_draw(
                cmd.command_buffer,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
    }
}

// Dynamic Rendering Recorder
pub struct DynamicRenderingRecorder<'a, 'b>(&'a CommandRecorder<'b>);

impl<'a, 'b> Deref for DynamicRenderingRecorder<'a, 'b>
where
    'b: 'a,
{
    type Target = CommandRecorder<'b>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a, 'b> DynamicRenderingRecorder<'a, 'b> {
    pub fn begin(command_recorder: &'a CommandRecorder<'b>, rendering_info: RenderingInfo) -> Self {
        let _ = unsafe {
            command_recorder
                .device
                .cmd_begin_rendering(command_recorder.command_buffer, &rendering_info)
        };
        Self(command_recorder)
    }
}

impl Drop for DynamicRenderingRecorder<'_, '_> {
    fn drop(&mut self) {
        unsafe { self.device.cmd_end_rendering(self.command_buffer) };
    }
}

impl<'a> GenericCommandRecorder<'a> for DynamicRenderingRecorder<'_, 'a> {
    fn command_recorder(&self) -> &CommandRecorder<'a> {
        &self.0
    }
}

impl<'a> RenderingRecorder<'a> for DynamicRenderingRecorder<'_, 'a> {}
