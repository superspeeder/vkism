use crate::render::RenderSystem;
use crate::render::descriptor::DescriptorSetLayout;
use ash::vk;
use ash::vk::{PipelineLayoutCreateFlags, PipelineLayoutCreateInfo, StructureType};
use std::marker::PhantomData;
use std::ops::Range;
use std::rc::Rc;

pub struct PipelineLayout {
    pipeline_layout: vk::PipelineLayout,
    device: ash::Device,
}

pub struct PipelineLayoutDescription {
    pub push_constant_ranges: Vec<(vk::ShaderStageFlags, Range<u32>)>,
    pub descriptor_set_layouts: Vec<Rc<DescriptorSetLayout>>,
}

impl PipelineLayout {
    pub fn new(
        render_system: &RenderSystem,
        description: &PipelineLayoutDescription,
    ) -> anyhow::Result<Self> {
        unsafe {
            let device = render_system.device().clone();
            let pcrs = description
                .push_constant_ranges
                .iter()
                .cloned()
                .map(|(stage, range)| {
                    vk::PushConstantRange::default()
                        .stage_flags(stage)
                        .offset(range.start)
                        .size(range.len() as u32)
                })
                .collect::<Vec<_>>();

            let descriptor_set_layouts = description
                .descriptor_set_layouts
                .iter()
                .map(|dsl| dsl.handle())
                .collect::<Vec<_>>();

            Ok(Self {
                pipeline_layout: device.create_pipeline_layout(
                    &PipelineLayoutCreateInfo {
                        s_type: StructureType::PIPELINE_LAYOUT_CREATE_INFO,
                        p_next: std::ptr::null_mut(),
                        flags: PipelineLayoutCreateFlags::empty(),
                        set_layout_count: descriptor_set_layouts.len() as u32,
                        p_set_layouts: descriptor_set_layouts.as_ptr(),
                        push_constant_range_count: pcrs.len() as u32,
                        p_push_constant_ranges: pcrs.as_ptr(),
                        _marker: PhantomData,
                    },
                    None,
                )?,
                device,
            })
        }
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe { self.device.destroy_pipeline_layout(self.pipeline_layout, None) };
    }
}

pub struct Pipeline {
    device: ash::Device,
    pipeline: ash::vk::Pipeline,
}

pub enum PipelineRenderCompatibility {
    UseRenderPass(vk::RenderPass), // TODO: if we abstract render passes, replace this with the abstraction
    RenderingInfo {
        view_mask: u32,
        color_attachment_formats: Vec<vk::Format>,
        depth_attachment_format: vk::Format,
        stencil_attachment_format: vk::Format,
    },
}

impl PipelineRenderCompatibility {
    pub fn simple_from_format(format: vk::Format) -> Self {
        Self::RenderingInfo {
            view_mask: 0,
            color_attachment_formats: vec![format],
            depth_attachment_format: vk::Format::UNDEFINED,
            stencil_attachment_format: vk::Format::UNDEFINED,
        }
    }

    pub fn simple_from_format_d24s8(format: vk::Format) -> Self {
        Self::RenderingInfo {
            view_mask: 0,
            color_attachment_formats: vec![format],
            depth_attachment_format: vk::Format::D24_UNORM_S8_UINT,
            stencil_attachment_format: vk::Format::D24_UNORM_S8_UINT,
        }
    }
}

pub struct PipelineDescription {
    pub layout: Rc<PipelineLayout>,
    pub rendering_compatibility: PipelineRenderCompatibility,
    // TODO: fixed function params
}
