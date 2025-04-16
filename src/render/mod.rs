pub mod command_buffer;
pub mod primary_renderer;
pub mod render_target;
pub mod pipeline;
pub mod descriptor;
pub mod shader;

use crate::render::command_buffer::CommandBuffer;
use crate::window::WindowSystem;
use anyhow::anyhow;
use ash::vk;
use ash::vk::{
    ApplicationInfo, CommandBufferAllocateInfo, CommandPoolCreateFlags, CommandPoolCreateInfo,
    DeviceCreateInfo, DeviceQueueCreateInfo, FenceCreateFlags, FenceCreateInfo, InstanceCreateInfo,
    SemaphoreCreateFlags, SemaphoreCreateInfo, StructureType,
};
use log::{debug, info};
use std::ffi::{CStr, CString, c_char};
use std::marker::PhantomData;

const DEVICE_EXTENSIONS: [&CStr; 1] = [ash::khr::swapchain::NAME];

pub struct QueueFamilyInfo {
    pub main: u32,
    pub present: u32,
    pub transfer: u32,
}

pub struct Queues {
    pub main: vk::Queue,
    pub present: vk::Queue,
    pub transfer: vk::Queue,
}

pub struct RenderSystem {
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue_family_info: QueueFamilyInfo,
    queues: Queues,
    main_pool: vk::CommandPool,
}

impl RenderSystem {
    pub fn new(window_system: &WindowSystem) -> anyhow::Result<Self> {
        let entry = unsafe { ash::Entry::load() }?;

        let application_info = ApplicationInfo::default().api_version(vk::API_VERSION_1_3);

        let required_extensions = window_system.get_required_instance_extensions().map_or(
            Err(anyhow!("Failed to query required instance extensions")),
            |e| Ok(e),
        )?;

        let required_extensions_cstrings = required_extensions
            .into_iter()
            .map(CString::new)
            .filter_map(Result::ok)
            .collect::<Vec<_>>();

        let required_extensions_cstrs = required_extensions_cstrings
            .iter()
            .map(|cs| cs.as_ptr())
            .collect::<Vec<_>>();

        let instance_create_info = InstanceCreateInfo::default()
            .application_info(&application_info)
            .enabled_extension_names(required_extensions_cstrs.as_slice());
        let instance = unsafe { entry.create_instance(&instance_create_info, None) }?;

        let physical_devices = unsafe { instance.enumerate_physical_devices() }?;
        let physical_device = physical_devices
            .iter()
            .next()
            .ok_or(anyhow!("No physical device available"))?
            .clone();

        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };
        info!(
            "Selected GPU: {}",
            CStr::from_bytes_until_nul(unsafe {
                std::mem::transmute::<_, &[u8; 256]>(&physical_device_properties.device_name)
            })?
            .to_str()?
        );

        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let mut main_queue: Option<u32> = None;
        let mut present_queue: Option<u32> = None;
        let mut transfer_queue: Option<u32> = None;

        for (index, properties) in queue_family_properties.iter().enumerate() {
            if main_queue.is_none() && properties.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                main_queue = Some(index as u32);
                if window_system.get_physical_device_presentation_support_raw(
                    instance.handle(),
                    physical_device,
                    index as u32,
                ) {
                    present_queue = Some(index as u32);
                }
            }

            if present_queue.is_none()
                && window_system.get_physical_device_presentation_support_raw(
                    instance.handle(),
                    physical_device,
                    index as u32,
                )
            {
                present_queue = Some(index as u32);
            }

            if transfer_queue.is_none()
                && !properties.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && !properties.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && properties.queue_flags.contains(vk::QueueFlags::TRANSFER)
            {
                transfer_queue = Some(index as u32);
            }
        }

        if main_queue.is_none() || present_queue.is_none() {
            return Err(anyhow!(
                "Missing required queue family support on targeted GPU."
            ));
        }

        if transfer_queue.is_none() {
            transfer_queue = Some(main_queue.unwrap());
            debug!("No exclusive transfer queue available, defaulting to main queue");
        }

        let queue_family_info = QueueFamilyInfo {
            main: main_queue.unwrap(),
            present: present_queue.unwrap(),
            transfer: transfer_queue.unwrap(),
        };

        let device_extensions = DEVICE_EXTENSIONS
            .iter()
            .cloned()
            .map(CStr::as_ptr)
            .collect::<Vec<*const c_char>>();

        const QUEUE_PRIORITIES: [f32; 1] = [1.0];

        let mut device_queue_create_infos: Vec<DeviceQueueCreateInfo> = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_priorities(&QUEUE_PRIORITIES)
                .queue_family_index(queue_family_info.main),
        ];

        if queue_family_info.present != queue_family_info.main {
            device_queue_create_infos.push(
                DeviceQueueCreateInfo::default()
                    .queue_priorities(&QUEUE_PRIORITIES)
                    .queue_family_index(queue_family_info.present),
            );
        }

        if queue_family_info.transfer != queue_family_info.main
            && queue_family_info.transfer != queue_family_info.present
        {
            device_queue_create_infos.push(
                DeviceQueueCreateInfo::default()
                    .queue_priorities(&QUEUE_PRIORITIES)
                    .queue_family_index(queue_family_info.transfer),
            );
        }

        let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .inline_uniform_block(true)
            .synchronization2(true);

        let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
            .descriptor_indexing(true)
            .draw_indirect_count(true)
            .runtime_descriptor_array(true)
            .timeline_semaphore(true);

        let mut vulkan_11_features =
            vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);

        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(
                vk::PhysicalDeviceFeatures::default()
                    .fill_mode_non_solid(true)
                    .tessellation_shader(true)
                    .geometry_shader(true)
                    .large_points(true)
                    .wide_lines(true)
                    .multi_draw_indirect(true),
            )
            .push_next(&mut vulkan_11_features)
            .push_next(&mut vulkan_12_features)
            .push_next(&mut vulkan_13_features);

        let device_create_info = DeviceCreateInfo::default()
            .enabled_extension_names(&device_extensions)
            .queue_create_infos(&device_queue_create_infos)
            .push_next(&mut features);

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }?;

        let queues = Queues {
            main: unsafe { device.get_device_queue(queue_family_info.main, 0) },
            present: unsafe { device.get_device_queue(queue_family_info.present, 0) },
            transfer: unsafe { device.get_device_queue(queue_family_info.transfer, 0) },
        };

        let main_pool = unsafe {
            device.create_command_pool(
                &CommandPoolCreateInfo {
                    s_type: StructureType::COMMAND_POOL_CREATE_INFO,
                    p_next: std::ptr::null_mut(),
                    flags: CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    queue_family_index: queue_family_info.main,
                    _marker: PhantomData,
                },
                None,
            )
        }?;

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            queue_family_info,
            queues,
            main_pool,
        })
    }

    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn create_semaphore(&self) -> anyhow::Result<vk::Semaphore> {
        const CREATE_INFO: SemaphoreCreateInfo = SemaphoreCreateInfo {
            s_type: StructureType::SEMAPHORE_CREATE_INFO,
            p_next: std::ptr::null_mut(),
            flags: SemaphoreCreateFlags::empty(),
            _marker: PhantomData,
        };

        unsafe { self.device.create_semaphore(&CREATE_INFO, None) }.map_err(anyhow::Error::from)
    }

    #[inline]
    pub fn create_fence(&self, signaled: bool) -> anyhow::Result<vk::Fence> {
        const CREATE_INFO_UNSIGNALED: FenceCreateInfo = FenceCreateInfo {
            s_type: StructureType::FENCE_CREATE_INFO,
            p_next: std::ptr::null_mut(),
            flags: FenceCreateFlags::empty(),
            _marker: PhantomData,
        };

        const CREATE_INFO_SIGNALED: FenceCreateInfo = FenceCreateInfo {
            s_type: StructureType::FENCE_CREATE_INFO,
            p_next: std::ptr::null_mut(),
            flags: FenceCreateFlags::SIGNALED,
            _marker: PhantomData,
        };

        if signaled {
            unsafe { self.device.create_fence(&CREATE_INFO_SIGNALED, None) }
        } else {
            unsafe { self.device.create_fence(&CREATE_INFO_UNSIGNALED, None) }
        }
        .map_err(anyhow::Error::from)
    }

    pub fn create_command_buffers<const N: usize>(
        &self,
        level: vk::CommandBufferLevel,
    ) -> anyhow::Result<[CommandBuffer; N]> {
        unsafe {
            self.device
                .allocate_command_buffers(&CommandBufferAllocateInfo {
                    s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                    p_next: std::ptr::null_mut(),
                    command_pool: self.main_pool,
                    level,
                    command_buffer_count: N as u32,
                    _marker: PhantomData,
                })
        }?
        .into_iter()
        .map(|c| CommandBuffer::wrap(&self.device, c))
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| anyhow!("Command buffer allocation failed"))
    }

    pub fn queues(&self) -> &Queues {
        &self.queues
    }

    pub fn queue_families(&self) -> &QueueFamilyInfo {
        &self.queue_family_info
    }

    pub fn pipeline_cache(&self) -> vk::PipelineCache {
        vk::PipelineCache::null() // TODO: impl this
    }
}

impl Drop for RenderSystem {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
