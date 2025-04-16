use std::{ops::{Div, Rem}, path::Path};

use ash::vk;
use log::warn;

use super::RenderSystem;

pub struct ShaderModule {
    shader_module: vk::ShaderModule,
    device: ash::Device,
}

impl ShaderModule {
    pub fn new(render_system: &RenderSystem, code: &[u32]) -> anyhow::Result<Self> {
        unsafe {
            let create_info = vk::ShaderModuleCreateInfo::default().code(code);
            let device = render_system.device().clone();
            Ok(Self {
                shader_module: device.create_shader_module(&create_info, None)?,
                device,
            })
        }
    }

    pub fn load_spv(render_system: &RenderSystem, path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let code_u8 = std::fs::read(path)?;
        let usable = code_u8.len().div(std::mem::size_of::<u32>());
        let code = unsafe { std::mem::transmute::<&[u8], &[u32]>(&code_u8[0..usable]) };

        Self::new(render_system, code)
    }

    pub fn load_glsl(render_system: &RenderSystem, path: impl AsRef<Path>, kind: shaderc::ShaderKind) -> anyhow::Result<Self> {
        let code = std::fs::read_to_string(path.as_ref())?;
        Self::load_glsl_from_memory_with_filename(render_system, code.as_str(), kind, path)
    }

    pub fn load_glsl_from_memory(render_system: &RenderSystem, code: &str, kind: shaderc::ShaderKind) -> anyhow::Result<Self> {
        Self::load_glsl_from_memory_with_filename(render_system, code, kind, "")
    }

    pub fn load_glsl_from_memory_with_filename(render_system: &RenderSystem, code: &str, kind: shaderc::ShaderKind, filename: impl AsRef<Path>) -> anyhow::Result<Self> {
        let filename = filename.as_ref().to_str().unwrap_or("");
        let compiler = shaderc::Compiler::new()?;
        let spirv = compiler.compile_into_spirv(code, kind, filename, "main", None)?;
        if spirv.get_num_warnings() > 0 {
            if filename.is_empty() {
                warn!("Warning while building shader: {}", spirv.get_warning_messages());
            } else {
                warn!("Warning while building shader '{}': {}", filename, spirv.get_warning_messages());
            }
        }
        
        Self::new(render_system, spirv.as_binary())
    }

}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe { self.device.destroy_shader_module(self.shader_module, None) };
    }
}


