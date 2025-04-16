use ash::vk;
use crate::render::RenderSystem;

pub struct DescriptorSetLayout {
    descriptor_set_layout: vk::DescriptorSetLayout,
    device: ash::Device,
}

pub struct DescriptorSetLayoutDescription<'a> {
    pub flags: vk::DescriptorSetLayoutCreateFlags,
    pub bindings: Vec<vk::DescriptorSetLayoutBinding<'a>>,
}

impl DescriptorSetLayout {
    pub fn new(render_system: &RenderSystem, description: &DescriptorSetLayoutDescription) -> anyhow::Result<Self> {
        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(description.bindings.as_slice())
            .flags(description.flags);

        Ok(unsafe {
            let device = render_system.device().clone();
            Self {
                descriptor_set_layout: device.create_descriptor_set_layout(&create_info, None)?,
                device,
            }
        })
    }

    pub fn handle(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe { self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
    }
}