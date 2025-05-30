use crate::render::RenderSystem;
use crate::render::descriptor::DescriptorSetLayout;
use crate::render::shader::ShaderModule;
use anyhow::anyhow;
use ash::vk;
use ash::vk::{CullModeFlags, FrontFace, Offset2D, PipelineLayoutCreateFlags, PipelineLayoutCreateInfo, PolygonMode, Rect2D, SampleCountFlags, ShaderStageFlags, StructureType, Viewport};
use std::ffi::CString;
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

    #[inline]
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

pub struct GraphicsPipeline {
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
    #[inline]
    pub fn simple_from_format(format: vk::Format) -> Self {
        Self::RenderingInfo {
            view_mask: 0,
            color_attachment_formats: vec![format],
            depth_attachment_format: vk::Format::UNDEFINED,
            stencil_attachment_format: vk::Format::UNDEFINED,
        }
    }

    #[inline]
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
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub clamp_depth: bool,
    pub discard_output: bool,
    pub depth_bias: Option<RasterizerDepthBias>,
    pub line_width: f32,
}

pub struct MultisamplingDescription {
    pub rasterization_samples: vk::SampleCountFlags,
    pub min_sample_shading: Option<f32>, // sample shading will be disabled if this is None
    pub sample_mask: [u32; 2],
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

pub struct GraphicsPipelineDescription {
    pub layout: Rc<PipelineLayout>,
    pub rendering_compatibility: PipelineRenderCompatibility,
    pub shader_stages: Vec<(vk::ShaderStageFlags, Rc<ShaderModule>, String)>,
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

impl GraphicsPipeline {
    pub fn new(
        render_system: &RenderSystem,
        description: &GraphicsPipelineDescription,
    ) -> anyhow::Result<Self> {
        let vertex_description = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(description.vertex_layout.bindings.as_slice())
            .vertex_attribute_descriptions(description.vertex_layout.attributes.as_slice());

        let (render_pass, subpass, mut rendering_state) = match &description.rendering_compatibility
        {
            PipelineRenderCompatibility::UseRenderPass(render_pass, subpass) => {
                (render_pass.clone(), subpass.clone(), None)
            }
            PipelineRenderCompatibility::RenderingInfo {
                view_mask,
                color_attachment_formats,
                depth_attachment_format,
                stencil_attachment_format,
            } => (
                vk::RenderPass::null(),
                0,
                Some(
                    vk::PipelineRenderingCreateInfo::default()
                        .color_attachment_formats(color_attachment_formats.as_slice())
                        .depth_attachment_format(depth_attachment_format.clone())
                        .stencil_attachment_format(stencil_attachment_format.clone())
                        .view_mask(view_mask.clone()),
                ),
            ),
        };

        let stages_mapped = description
            .shader_stages
            .iter()
            .cloned()
            .map(|(stage, module, entry)| (stage, module, CString::new(entry).unwrap()))
            .collect::<Vec<_>>();

        let stages = stages_mapped
            .iter()
            .map(|(stage, module, entry)| {
                vk::PipelineShaderStageCreateInfo::default()
                    .module(module.handle())
                    .stage(stage.clone())
                    .name(entry.as_c_str())
            })
            .collect::<Vec<_>>();

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .primitive_restart_enable(description.allow_primitive_restart)
            .topology(description.primitive_topology);

        let tessellation_state = vk::PipelineTessellationStateCreateInfo::default()
            .patch_control_points(description.tessellator_patch_control_points);

        let (viewports, scissors) = description.viewports.iter().cloned().fold(
            (Vec::<vk::Viewport>::new(), Vec::<vk::Rect2D>::new()),
            |(mut viewports, mut scissors), (viewport, scissor)| {
                viewports.push(viewport);
                scissors.push(scissor);
                (viewports, scissors)
            },
        );

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(viewports.as_slice())
            .scissors(scissors.as_slice());

        let mut rasterizer_state = vk::PipelineRasterizationStateCreateInfo::default()
            .cull_mode(description.rasterizer.cull_mode)
            .polygon_mode(description.rasterizer.polygon_mode)
            .depth_clamp_enable(description.rasterizer.clamp_depth)
            .depth_bias_enable(description.rasterizer.depth_bias.is_some())
            .line_width(description.rasterizer.line_width)
            .front_face(description.rasterizer.front_face)
            .rasterizer_discard_enable(description.rasterizer.discard_output);
        if let Some(bias) = description.rasterizer.depth_bias.as_ref() {
            rasterizer_state = rasterizer_state
                .depth_bias_clamp(bias.clamp)
                .depth_bias_constant_factor(bias.constant_factor)
                .depth_bias_slope_factor(bias.slope_facator);
        }

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .alpha_to_coverage_enable(description.multisampling.alpha_to_coverage)
            .alpha_to_one_enable(description.multisampling.alpha_to_one)
            .rasterization_samples(description.multisampling.rasterization_samples)
            .min_sample_shading(description.multisampling.min_sample_shading.unwrap_or(1.0))
            .sample_shading_enable(description.multisampling.min_sample_shading.is_some())
            .sample_mask(description.multisampling.sample_mask.as_slice());

        let mut depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default();

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(description.color_blending.logic_op.is_some())
            .logic_op(
                description
                    .color_blending
                    .logic_op
                    .unwrap_or(vk::LogicOp::CLEAR),
            )
            .blend_constants(description.color_blending.blend_constants)
            .attachments(description.color_blending.attachments.as_slice());

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(description.dynamic_states.as_slice());

        let mut create_info = vk::GraphicsPipelineCreateInfo::default()
            .layout(description.layout.handle())
            .render_pass(render_pass)
            .subpass(subpass)
            .stages(stages.as_slice())
            .vertex_input_state(&vertex_description)
            .input_assembly_state(&input_assembly_state)
            .tessellation_state(&tessellation_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state);

        if let Some(depth_stencil) = &description.depth_stencil {
            depth_stencil_state = depth_stencil_state
                .depth_test_enable(depth_stencil.depth_test)
                .depth_write_enable(depth_stencil.depth_write)
                .depth_compare_op(depth_stencil.depth_compare_op)
                .depth_bounds_test_enable(depth_stencil.depth_bounds_test)
                .stencil_test_enable(depth_stencil.stencil_test)
                .front(depth_stencil.stencil_front)
                .back(depth_stencil.stencil_back)
                .min_depth_bounds(depth_stencil.depth_bounds.start)
                .max_depth_bounds(depth_stencil.depth_bounds.end);
            create_info = create_info.depth_stencil_state(&depth_stencil_state);
        }

        if let Some(rendering_state) = rendering_state.as_mut() {
            create_info = create_info.push_next(rendering_state);
        }

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

    #[inline]
    pub fn handle(&self) -> vk::Pipeline {
        self.pipeline
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

#[inline]
pub const fn standard_blend_attachment() -> vk::PipelineColorBlendAttachmentState {
    vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::TRUE,
        src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::RGBA,
    }
}

impl Default for RasterizerDescription {
    fn default() -> Self {
        Self {
            polygon_mode: PolygonMode::FILL,
            cull_mode: CullModeFlags::NONE,
            front_face: FrontFace::CLOCKWISE,
            clamp_depth: false,
            discard_output: false,
            depth_bias: None,
            line_width: 1.0,
        }
    }
}

impl Default for ColorBlendingDescription {
    fn default() -> Self {
        Self {
            logic_op: None,
            blend_constants: [0.0, 0.0, 0.0, 0.0],
            attachments: vec![],
        }
    }
}

impl Default for MultisamplingDescription {
    fn default() -> Self {
        Self {
            rasterization_samples: SampleCountFlags::TYPE_1,
            min_sample_shading: None,
            sample_mask: [!0u32, !0u32],
            alpha_to_coverage: false,
            alpha_to_one: false,
        }
    }
}

#[inline]
pub const fn standard_viewport_scissor_from_extent(extent: vk::Extent2D) -> (Viewport, Rect2D) {
    (
        Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        },
        Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent,
        },
    )
}

impl Default for VertexLayout {
    fn default() -> Self {
        Self {
            bindings: vec![],
            attributes: vec![],
        }
    }
}

#[inline]
pub fn standard_vertex_fragment_stages(vertex_shader: Rc<ShaderModule>, fragment_shader: Rc<ShaderModule>) -> Vec<(ShaderStageFlags, Rc<ShaderModule>, String)> {
    vec![
        (
            ShaderStageFlags::VERTEX,
            vertex_shader.clone(),
            "main".to_string(),
        ),
        (
            ShaderStageFlags::FRAGMENT,
            fragment_shader.clone(),
            "main".to_string(),
        ),
    ]
}


