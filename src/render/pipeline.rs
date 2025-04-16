use crate::render::RenderSystem;
use crate::render::descriptor::DescriptorSetLayout;
use crate::render::shader::ShaderModule;
use anyhow::anyhow;
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

    pub fn handle(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None)
        };
    }
}

pub struct Pipeline {
    pipeline: ash::vk::Pipeline,
    device: ash::Device,
}

pub enum PipelineRenderCompatibility {
    UseRenderPass(vk::RenderPass, u32), // TODO: if we abstract render passes, replace this with the abstraction
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

pub struct VertexLayout {
    pub bindings: Vec<vk::VertexInputBindingDescription>,
    pub attributes: Vec<vk::VertexInputAttributeDescription>,
}

pub struct RasterizerDepthBias {
    pub constant_factor: f32,
    pub slope_facator: f32,
    pub clamp: f32,
}

pub struct RasterizerDescription {
    pub polygon_mode: vk::PolygonMode,
    pub clamp_depth: bool,
    pub discard_output: bool,
    pub depth_bias: Option<RasterizerDepthBias>,
    pub line_width: f32,
}

pub struct MultisamplingDescription {
    pub rasterization_samples: vk::SampleCountFlags,
    pub min_sample_shading: Option<f32>, // sample shading will be disabled if this is None
    pub sample_mask: u64, // will be interpreted based on size of rasterization_samples
    pub alpha_to_coverage: bool,
    pub alpha_to_one: bool,
}

pub struct DepthStencilDescription {
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_compare_op: vk::CompareOp,
    pub depth_bounds_test: bool,

    pub stencil_test: bool,
    pub stencil_front: vk::StencilOpState,
    pub stencil_back: vk::StencilOpState,
    pub depth_bounds: Range<f32>,
}

pub struct ColorBlendingDescription {
    pub logic_op: Option<vk::LogicOp>,
    pub blend_constants: [f32; 4],
    pub attachments: Vec<vk::PipelineColorBlendAttachmentState>,
}

pub struct PipelineDescription {
    pub layout: Rc<PipelineLayout>,
    pub rendering_compatibility: PipelineRenderCompatibility,
    pub shader_stages: (vk::ShaderStageFlags, Rc<ShaderModule>, String),
    pub vertex_layout: VertexLayout,

    // fixed function
    pub primitive_topology: vk::PrimitiveTopology,
    pub allow_primitive_restart: bool,
    pub tessellator_patch_control_points: u32,
    pub viewports: Vec<(vk::Viewport, vk::Rect2D)>,
    pub rasterizer: RasterizerDescription,
    pub multisampling: MultisamplingDescription,
    pub depth_stencil: Option<DepthStencilDescription>,
    pub color_blending: ColorBlendingDescription,
    pub dynamic_states: Vec<vk::DynamicState>,
}

impl Pipeline {
    pub fn new(
        render_system: &RenderSystem,
        description: &PipelineDescription,
    ) -> anyhow::Result<Self> {
        let vertex_description = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(description.vertex_layout.bindings.as_slice());

        let mut create_info =
            vk::GraphicsPipelineCreateInfo::default().layout(description.layout.handle());

        let device = render_system.device().clone();
        unsafe {
            Ok(Self {
                pipeline: device
                    .create_graphics_pipelines(render_system.pipeline_cache(), &[create_info], None)
                    .map_err(|(_, e)| e)?
                    .first()
                    .cloned()
                    .ok_or(anyhow!("Failed to create graphics pipeline"))?,
                device,
            })
        }
    }
}
