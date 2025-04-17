use crate::render::RenderSystem;
use ash::vk;
use ash::vk::{
    DescriptorPoolCreateFlags, DescriptorPoolResetFlags, DescriptorSet, DescriptorSetAllocateInfo,
    StructureType,
};
use std::marker::PhantomData;
use std::rc::Rc;

pub struct DescriptorSetLayout {
    descriptor_set_layout: vk::DescriptorSetLayout,
    device: ash::Device,
}

pub struct DescriptorSetLayoutDescription<'a> {
    pub flags: vk::DescriptorSetLayoutCreateFlags,
    pub bindings: Vec<vk::DescriptorSetLayoutBinding<'a>>,
}

impl DescriptorSetLayout {
    pub fn new(
        render_system: &RenderSystem,
        description: &DescriptorSetLayoutDescription,
    ) -> anyhow::Result<Self> {
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

    #[inline]
    pub fn handle(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None)
        };
    }
}

pub struct DescriptorPool {
    descriptor_pool: vk::DescriptorPool,
    device: ash::Device,
}

impl DescriptorPool {
    pub fn new(
        render_system: &RenderSystem,
        max_sets: u32,
        pool_sizes: &[(vk::DescriptorType, u32)],
        flags: DescriptorPoolCreateFlags,
    ) -> anyhow::Result<Self> {
        let device = render_system.device().clone();
        let pool_sizes = pool_sizes
            .iter()
            .map(|&(ty, descriptor_count)| vk::DescriptorPoolSize {
                ty,
                descriptor_count,
            })
            .collect::<Vec<_>>();

        unsafe {
            Ok(Self {
                descriptor_pool: device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo {
                        s_type: StructureType::DESCRIPTOR_POOL_CREATE_INFO,
                        p_next: std::ptr::null_mut(),
                        flags,
                        max_sets,
                        pool_size_count: pool_sizes.len() as u32,
                        p_pool_sizes: pool_sizes.as_ptr(),
                        _marker: PhantomData,
                    },
                    None,
                )?,
                device,
            })
        }
    }

    pub fn allocate_descriptor_sets(
        &self,
        descriptor_set_layouts: &[Rc<DescriptorSetLayout>],
    ) -> anyhow::Result<Vec<DescriptorSet>> {
        let descriptor_set_layouts = descriptor_set_layouts
            .iter()
            .map(|dsl| dsl.descriptor_set_layout)
            .collect::<Vec<_>>();

        unsafe {
            Ok(self
                .device
                .allocate_descriptor_sets(&DescriptorSetAllocateInfo {
                    s_type: StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                    p_next: std::ptr::null_mut(),
                    descriptor_pool: self.descriptor_pool,
                    descriptor_set_count: descriptor_set_layouts.len() as u32,
                    p_set_layouts: descriptor_set_layouts.as_ptr(),
                    _marker: PhantomData,
                })?)
        }
    }

    pub fn free_descriptor_sets(&self, descriptor_sets: &[DescriptorSet]) -> anyhow::Result<()> {
        unsafe {
            Ok(self
                .device
                .free_descriptor_sets(self.descriptor_pool, descriptor_sets)?)
        }
    }

    pub fn reset(&self) -> anyhow::Result<()> {
        unsafe {
            Ok(self
                .device
                .reset_descriptor_pool(self.descriptor_pool, DescriptorPoolResetFlags::empty())?)
        }
    }


}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
