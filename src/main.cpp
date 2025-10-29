#include "vulkan/vulkan.hpp"
#include <cstdint>
#include <iostream>
#include <fstream>
#include <ranges>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <limits>
#include <array>
#include <chrono>
#include <assert.h>
#include <vulkan/vulkan_core.h>

// WTF
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include "../vendor/glfw/include/GLFW/glfw3.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
// #include <glm/glm.hpp>
// #include <glm/gtc/matrix_transform.hpp>
// #include <glm/gtx/hash.hpp>
#include "../vendor/glm/glm/glm.hpp"
#include "../vendor/glm/glm/gtc/matrix_transform.hpp"
#include "../vendor/glm/glm/gtx/hash.hpp"

#define STB_IMAGE_IMPLEMENTATION
// #include <stb/stb_image.h>
#include "../vendor/stb/stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION
// #include <tinyobjloader/tiny_obj_loader.h>
#include "../vendor/tinyobjloader/tiny_obj_loader.h"

#define LOG_FUNCTIONS 1
#if LOG_FUNCTIONS
#define LOG_FUNCTION() \
    printf("Function %s() called in %s at line %d\n", __func__, __FILE__, __LINE__);
#else
#define LOG_FUNCTION()
#endif

#define ASSERT(x) (assert(x))

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<char const*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return { 0, sizeof(Vertex), vk::VertexInputRate::eVertex };
    }

    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription( 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos) ),
            vk::VertexInputAttributeDescription( 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color) ),
            vk::VertexInputAttributeDescription( 2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord) )
        };
    }

    bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
    }
};

template<> struct std::hash<Vertex> {
    size_t operator()(Vertex const& vertex) const noexcept {
        return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
    }
};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

class vk_context {
public:
    void run();
    ~vk_context();
private:
    GLFWwindow *                     window = nullptr;
    vk::raii::Context                context;
    vk::raii::Instance               instance       = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::SurfaceKHR             surface        = nullptr;
    vk::raii::PhysicalDevice         physicalDevice = nullptr;
    vk::raii::Device                 device         = nullptr;
    uint32_t                         queueIndex     = ~0;
    vk::raii::Queue                  queue          = nullptr;
    vk::raii::SwapchainKHR           swapChain      = nullptr;
    std::vector<vk::Image>           swapChainImages;
    vk::SurfaceFormatKHR             swapChainSurfaceFormat;
    vk::Extent2D                     swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    vk::raii::Image depthImage = nullptr;
    vk::raii::DeviceMemory depthImageMemory = nullptr;
    vk::raii::ImageView depthImageView = nullptr;

    vk::raii::Image textureImage = nullptr;
    vk::raii::DeviceMemory textureImageMemory = nullptr;
    vk::raii::ImageView textureImageView = nullptr;
    vk::raii::Sampler textureSampler = nullptr;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    vk::raii::Buffer vertexBuffer = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;
    vk::raii::Buffer indexBuffer = nullptr;
    vk::raii::DeviceMemory indexBufferMemory = nullptr;

    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;

    std::vector<vk::raii::Semaphore> presentCompleteSemaphore;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphore;
    std::vector<vk::raii::Fence> inFlightFences;
    uint32_t semaphoreIndex = 0;
    uint32_t currentFrame = 0;

    bool framebufferResized = false;

    std::vector<const char*> requiredDeviceExtension = {
        vk::KHRSwapchainExtensionName,
        vk::KHRSpirv14ExtensionName,
        vk::KHRSynchronization2ExtensionName,
        vk::KHRCreateRenderpass2ExtensionName,
    };

    void initWindow();

    /* TODO: Are u a member? */
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    void mainLoop();
    void cleanupSwapChain();
    void cleanup() const;
    void createSwapChain();
    void recreateSwapChain();
    void createInstance();
    void createLogicalDevice();
    void createImageViews();
    void createDepthResources();
    void drawFrame();
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const;
    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) const;
    vk::Format findDepthFormat() const;
    void createCommandPool();
    vk::raii::ImageView createImageView(vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags);
    void createImage(uint32_t width,
                     uint32_t height,
                     vk::Format format,
                     vk::ImageTiling tiling,
                     vk::ImageUsageFlags usage,
                     vk::MemoryPropertyFlags properties,
                     vk::raii::Image& image,
                     vk::raii::DeviceMemory& imageMemory);
    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
    void createTextureImage();
    void createTextureImageView();
    void createTextureSampler();
    void createBuffer(vk::DeviceSize size,
                      vk::BufferUsageFlags usage,
                      vk::MemoryPropertyFlags properties,
                      vk::raii::Buffer& buffer,
                      vk::raii::DeviceMemory& bufferMemory);
    std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands();
    void endSingleTimeCommands(const vk::raii::CommandBuffer& commandBuffer) const;
    void copyBuffer(vk::raii::Buffer & srcBuffer, vk::raii::Buffer & dstBuffer, vk::DeviceSize size);
    void transitionImageLayout(const vk::raii::Image& image,
                               vk::ImageLayout oldLayout,
                               vk::ImageLayout newLayout);
    void copyBufferToImage(const vk::raii::Buffer& buffer,
                           vk::raii::Image& image,
                           uint32_t width,
                           uint32_t height);
    void loadModel();
    void createVertexBuffer();
    void createIndexBuffer();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffers();
    void recordCommandBuffer(uint32_t imageIndex);
    void transition_image_layout(
        uint32_t imageIndex,
        vk::ImageLayout old_layout,
        vk::ImageLayout new_layout,
        vk::AccessFlags2 src_access_mask,
        vk::AccessFlags2 dst_access_mask,
        vk::PipelineStageFlags2 src_stage_mask,
        vk::PipelineStageFlags2 dst_stage_mask);
    void createSyncObjects();
    void initVulkan();
    void updateUniformBuffer(uint32_t currentImage) const;

};

void vk_context::transitionImageLayout(const vk::raii::Image& image,
                           vk::ImageLayout oldLayout,
                           vk::ImageLayout newLayout)
{
    auto commandBuffer = beginSingleTimeCommands();

    vk::ImageMemoryBarrier barrier = {};
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.image = image;
    barrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.srcAccessMask =  vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask =  vk::AccessFlagBits::eShaderRead;

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else {
        throw std::invalid_argument("unsupported layout transition!");
    }
    commandBuffer->pipelineBarrier( sourceStage, destinationStage, {}, {}, nullptr, barrier );
    endSingleTimeCommands(*commandBuffer);
}

static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const & surfaceCapabilities)
{
    auto minImageCount = std::max( 3u, surfaceCapabilities.minImageCount );
    if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount)) {
        minImageCount = surfaceCapabilities.maxImageCount;
    }
    return minImageCount;
}

static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
    assert(!availableFormats.empty());
    const auto formatIt = std::ranges::find_if(
        availableFormats,
        []( const auto & format ) { return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear; } );
    return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
}

static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
    assert(std::ranges::any_of(availablePresentModes, [](auto presentMode){ return presentMode == vk::PresentModeKHR::eFifo; }));
    return std::ranges::any_of(availablePresentModes,
        [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; } ) ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
}

[[nodiscard]] std::vector<const char*> getRequiredExtensions()
{
    uint32_t glfwExtensionCount = 0;
    auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    if (enableValidationLayers) {
        extensions.push_back(vk::EXTDebugUtilsExtensionName );
    }

    return extensions;
}

static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
    if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError || severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
        std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
    }

    return vk::False;
}

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }
    std::vector<char> buffer(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    file.close();
    return buffer;
}

void vk_context::initWindow()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void vk_context::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    /* TODO: Does this matter, or can i make the compiler stop bitching about not being used */
    (void)width;
    (void)height;

    auto app = static_cast<vk_context*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}

void vk_context::mainLoop()
{
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrame();
    }

    device.waitIdle();
}

void vk_context::cleanupSwapChain()
{
    swapChainImageViews.clear();
    swapChain = nullptr;
}

void vk_context::cleanup() const
{
    LOG_FUNCTION()
    ASSERT(window != NULL);

    glfwDestroyWindow(window);

    glfwTerminate();
}

void vk_context::createSwapChain()
{
    auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    swapChainExtent          = chooseSwapExtent(surfaceCapabilities);
    swapChainSurfaceFormat   = chooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR(*surface));
    vk::SwapchainCreateInfoKHR swapChainCreateInfo = {};
    swapChainCreateInfo.surface          = *surface;
    swapChainCreateInfo.minImageCount    = chooseSwapMinImageCount(surfaceCapabilities);
    swapChainCreateInfo.imageFormat      = swapChainSurfaceFormat.format;
    swapChainCreateInfo.imageColorSpace  = swapChainSurfaceFormat.colorSpace;
    swapChainCreateInfo.imageExtent      = swapChainExtent;
    swapChainCreateInfo.imageArrayLayers = 1;
    swapChainCreateInfo.imageUsage       = vk::ImageUsageFlagBits::eColorAttachment;
    swapChainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
    swapChainCreateInfo.preTransform     = surfaceCapabilities.currentTransform;
    swapChainCreateInfo.compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    swapChainCreateInfo.presentMode      = chooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(*surface));
    swapChainCreateInfo.clipped          = true;

    swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
    swapChainImages = swapChain.getImages();
}

void vk_context::recreateSwapChain()
{
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    device.waitIdle();

    cleanupSwapChain();
    createSwapChain();
    createImageViews();
    createDepthResources();
}

void vk_context::createInstance()
{
    // WTF is this shit and why
    constexpr vk::ApplicationInfo appInfo(
        "Hello Triangle",                     // pApplicationName
        VK_MAKE_VERSION(1, 0, 0),             // applicationVersion
        "No Engine",                          // pEngineName
        VK_MAKE_VERSION(1, 0, 0),             // engineVersion
        vk::ApiVersion14                      // apiVersion
    );

    // Get the required layers
    std::vector<char const*> requiredLayers;
    if (enableValidationLayers) {
      requiredLayers.assign(validationLayers.begin(), validationLayers.end());
    }

    // Check if the required layers are supported by the Vulkan implementation.
    auto layerProperties = context.enumerateInstanceLayerProperties();
    for (auto const& requiredLayer : requiredLayers)
    {
        if (std::ranges::none_of(layerProperties,
                                 [requiredLayer](auto const& layerProperty)
                                 { return strcmp(layerProperty.layerName, requiredLayer) == 0; }))
        {
            throw std::runtime_error("Required layer not supported: " + std::string(requiredLayer));
        }
    }

    // Get the required extensions.
    auto requiredExtensions = getRequiredExtensions();

    // Check if the required extensions are supported by the Vulkan implementation.
    auto extensionProperties = context.enumerateInstanceExtensionProperties();
    for (auto const& requiredExtension : requiredExtensions)
    {
        if (std::ranges::none_of(extensionProperties,
                                 [requiredExtension](auto const& extensionProperty)
                                 { return strcmp(extensionProperty.extensionName, requiredExtension) == 0; }))
        {
            throw std::runtime_error("Required extension not supported: " + std::string(requiredExtension));
        }
    }

    vk::InstanceCreateInfo createInfo ={};
    createInfo.pApplicationInfo        = &appInfo;
    createInfo.enabledLayerCount       = static_cast<uint32_t>(requiredLayers.size());
    createInfo.ppEnabledLayerNames     = requiredLayers.data();
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(requiredExtensions.size());
    createInfo.ppEnabledExtensionNames = requiredExtensions.data();
    
    instance = vk::raii::Instance(context, createInfo);
}

void vk_context::setupDebugMessenger()
{
    if (!enableValidationLayers) return;

    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags( vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError );
    vk::DebugUtilsMessageTypeFlagsEXT    messageTypeFlags( vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation );

    vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT = {};
    debugUtilsMessengerCreateInfoEXT.messageSeverity = severityFlags;
    debugUtilsMessengerCreateInfoEXT.messageType = messageTypeFlags;
    debugUtilsMessengerCreateInfoEXT.pfnUserCallback = &debugCallback;

    debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
}

void vk_context::createSurface()
{
    VkSurfaceKHR       _surface;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
        throw std::runtime_error("failed to create window surface!");
    }
    surface = vk::raii::SurfaceKHR(instance, _surface);
}

void vk_context::pickPhysicalDevice()
{
    std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    const auto                            devIter = std::ranges::find_if(
      devices,
      [&]( auto const & device )
      {
        // Check if the device supports the Vulkan 1.3 API version
        bool supportsVulkan1_3 = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

        // Check if any of the queue families support graphics operations
        auto queueFamilies = device.getQueueFamilyProperties();
        bool supportsGraphics =
          std::ranges::any_of( queueFamilies, []( auto const & qfp ) { return !!( qfp.queueFlags & vk::QueueFlagBits::eGraphics ); } );

        // Check if all required device extensions are available
        auto availableDeviceExtensions = device.enumerateDeviceExtensionProperties();
        bool supportsAllRequiredExtensions =
          std::ranges::all_of( requiredDeviceExtension,
                               [&availableDeviceExtensions]( auto const & requiredDeviceExtension )
                               {
                                 return std::ranges::any_of( availableDeviceExtensions,
                                                             [requiredDeviceExtension]( auto const & availableDeviceExtension )
                                                             { return strcmp( availableDeviceExtension.extensionName, requiredDeviceExtension ) == 0; } );
                               } );

        auto features = device.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
        bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
                                        features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                        features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

        return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
      } );
    if ( devIter != devices.end() )
    {
        physicalDevice = *devIter;
    }
    else
    {
        throw std::runtime_error( "failed to find a suitable GPU!" );
    }
}

void vk_context::createLogicalDevice()
{
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    // get the first index into queueFamilyProperties which supports both graphics and present
    for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++)
    {
        if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
            physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface))
        {
            // found a queue family that supports both graphics and present
            queueIndex = qfpIndex;
            break;
        }
    }
    /* TODO: May cast ~0 to uint32_t, see if this fail somehow, i have done it to stop the warning */
    if (queueIndex == (uint32_t)~0) 
    {
        /* FIXME: throw is exceptions shit */
        throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
    }

    vk::PhysicalDeviceFeatures2 physicalDeviceFeatures2 = {};
    physicalDeviceFeatures2.features.samplerAnisotropy = VK_TRUE;

    vk::PhysicalDeviceVulkan13Features vk13_features = {};
    vk13_features.synchronization2 = VK_TRUE;
    vk13_features.dynamicRendering = VK_TRUE;

    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT extended_features = {};
    extended_features.extendedDynamicState = VK_TRUE;

    // query for Vulkan 1.3 features
    // WTF is this shit and why
    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features,
                       vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
    featureChain(
            physicalDeviceFeatures2,
            vk13_features,
            extended_features
    );

    // create a Device
    float queuePriority = 0.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo = {};
    deviceQueueCreateInfo.queueFamilyIndex = queueIndex;
    deviceQueueCreateInfo.queueCount = 1;
    deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

    vk::DeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>();
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size());
    deviceCreateInfo.ppEnabledExtensionNames = requiredDeviceExtension.data();

    device = vk::raii::Device( physicalDevice, deviceCreateInfo );
    queue = vk::raii::Queue( device, queueIndex, 0 );
}

void vk_context::createImageViews()
{
    assert(swapChainImageViews.empty());

    vk::ImageViewCreateInfo imageViewCreateInfo = {};
    imageViewCreateInfo.viewType = vk::ImageViewType::e2D;
    imageViewCreateInfo.format = swapChainSurfaceFormat.format;
    imageViewCreateInfo.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

    for ( auto image : swapChainImages )
    {
        imageViewCreateInfo.image = image;
        swapChainImageViews.emplace_back( device, imageViewCreateInfo );
    }
}

void vk_context::createDescriptorSetLayout()
{
    std::array bindings = {
        vk::DescriptorSetLayoutBinding( 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr),
        vk::DescriptorSetLayoutBinding( 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr)
    };

    vk::DescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
}

void vk_context::createGraphicsPipeline()
{
    vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo ={};
    vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
    vertShaderStageInfo.module = shaderModule;
    vertShaderStageInfo.pName = "vertMain";
    
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo = {};
    fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
    fragShaderStageInfo.module = shaderModule;
    fragShaderStageInfo.pName = "fragMain";

    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    vk::VertexInputBindingDescription bindingDescription = Vertex::getBindingDescription();
    std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions = Vertex::getAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssembly.primitiveRestartEnable = vk::False;

    vk::PipelineViewportStateCreateInfo viewportState = {};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    vk::PipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.depthClampEnable = vk::False;
    rasterizer.rasterizerDiscardEnable = vk::False;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = vk::False;
    rasterizer.lineWidth = 1.0f;

    vk::PipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
    multisampling.sampleShadingEnable = vk::False;

    vk::PipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.depthTestEnable = vk::True;
    depthStencil.depthWriteEnable = vk::True;
    depthStencil.depthCompareOp = vk::CompareOp::eLess;
    depthStencil.depthBoundsTestEnable = vk::False;
    depthStencil.stencilTestEnable = vk::False;

    vk::PipelineColorBlendAttachmentState colorBlendAttachment;
    colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = vk::False;

    vk::PipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.logicOpEnable = vk::False;
    colorBlending.logicOp = vk::LogicOp::eCopy;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    std::vector dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };
    vk::PipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &*descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;

    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    vk::Format depthFormat = findDepthFormat();

    vk::GraphicsPipelineCreateInfo graphicsPipelineCreateInfo = {};
    graphicsPipelineCreateInfo.stageCount = 2;
    graphicsPipelineCreateInfo.pStages = shaderStages;
    graphicsPipelineCreateInfo.pVertexInputState = &vertexInputInfo;
    graphicsPipelineCreateInfo.pInputAssemblyState = &inputAssembly;
    graphicsPipelineCreateInfo.pViewportState = &viewportState;
    graphicsPipelineCreateInfo.pRasterizationState = &rasterizer;
    graphicsPipelineCreateInfo.pMultisampleState = &multisampling;
    graphicsPipelineCreateInfo.pDepthStencilState = &depthStencil;
    graphicsPipelineCreateInfo.pColorBlendState = &colorBlending;
    graphicsPipelineCreateInfo.pDynamicState = &dynamicState;
    graphicsPipelineCreateInfo.layout = pipelineLayout;
    graphicsPipelineCreateInfo.renderPass = nullptr;

    vk::PipelineRenderingCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.colorAttachmentCount = 1;
    pipelineCreateInfo.pColorAttachmentFormats = &swapChainSurfaceFormat.format;
    pipelineCreateInfo.depthAttachmentFormat = depthFormat;

    vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
        graphicsPipelineCreateInfo,
        pipelineCreateInfo
    };

    graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
}

void vk_context::createCommandPool()
{
    vk::CommandPoolCreateInfo poolInfo= {};
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    poolInfo.queueFamilyIndex = queueIndex;

    commandPool = vk::raii::CommandPool(device, poolInfo);
}

void vk_context::createDepthResources()
{
    vk::Format depthFormat = findDepthFormat();

    createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage, depthImageMemory);
    depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);
}

vk::Format vk_context::findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) const {
    for (const auto format : candidates) {
        vk::FormatProperties props = physicalDevice.getFormatProperties(format);

        if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
            return format;
        }
        if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}

[[nodiscard]] vk::Format vk_context::findDepthFormat() const {
    return findSupportedFormat(
    {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment
    );
}

static bool hasStencilComponent(vk::Format format) {
    return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
}

void vk_context::createTextureImage()
{
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    vk::DeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }

    vk::raii::Buffer stagingBuffer({});
    vk::raii::DeviceMemory stagingBufferMemory({});
    createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void* data = stagingBufferMemory.mapMemory(0, imageSize);
    memcpy(data, pixels, imageSize);
    stagingBufferMemory.unmapMemory();

    stbi_image_free(pixels);

    createImage(texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);

    transitionImageLayout(textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
    transitionImageLayout(textureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void vk_context::createTextureImageView()
{
    textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor);
}

void vk_context::createTextureSampler()
{
    vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
    vk::SamplerCreateInfo samplerInfo = {};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.anisotropyEnable = vk::True;
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.compareEnable = vk::False;
    samplerInfo.compareOp = vk::CompareOp::eAlways;
    textureSampler = vk::raii::Sampler(device, samplerInfo);
}

vk::raii::ImageView vk_context::createImageView(vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags)
{
    vk::ImageViewCreateInfo viewInfo = {};
    viewInfo.image = image,
    viewInfo.viewType = vk::ImageViewType::e2D,
    viewInfo.format = format,
    viewInfo.subresourceRange = { aspectFlags, 0, 1, 0, 1 };
    return vk::raii::ImageView(device, viewInfo);
}

void vk_context::createImage(uint32_t width,
                 uint32_t height,
                 vk::Format format,
                 vk::ImageTiling tiling,
                 vk::ImageUsageFlags usage,
                 vk::MemoryPropertyFlags properties,
                 vk::raii::Image& image,
                 vk::raii::DeviceMemory& imageMemory)
{
    vk::ImageCreateInfo imageInfo = {};
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.format = format;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    imageInfo.tiling = tiling;
    imageInfo.usage = usage;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;

    image = vk::raii::Image(device, imageInfo);

    vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo = {};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    imageMemory = vk::raii::DeviceMemory(device, allocInfo);
    image.bindMemory(imageMemory, 0);
}

void vk_context::copyBufferToImage(const vk::raii::Buffer& buffer,
                       vk::raii::Image& image,
                       uint32_t width,
                       uint32_t height)
{
    std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = beginSingleTimeCommands();
    vk::BufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
    region.imageOffset.x = 0;
    region.imageOffset.y = 0;
    region.imageOffset.z = 0;
    region.imageExtent.width = width;
    region.imageExtent.height = height;
    /* TODO: No harcode this, maybe??? */
    region.imageExtent.depth = 1;

    commandBuffer->copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, {region});
    endSingleTimeCommands(*commandBuffer);
}

void vk_context::loadModel()
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
        throw std::runtime_error(warn + err);
    }

    std::unordered_map<Vertex, uint32_t> uniqueVertices{};

    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            Vertex vertex{};

            vertex.pos = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]
            };

            vertex.texCoord = {
                attrib.texcoords[2 * index.texcoord_index + 0],
                1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
            };

            vertex.color = {1.0f, 1.0f, 1.0f};

            if (!uniqueVertices.contains(vertex)) {
                uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                vertices.push_back(vertex);
            }

            indices.push_back(uniqueVertices[vertex]);
        }
    }
}

void vk_context::createVertexBuffer()
{
    vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
    vk::raii::Buffer stagingBuffer({});
    vk::raii::DeviceMemory stagingBufferMemory({});
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(dataStaging, vertices.data(), bufferSize);
    stagingBufferMemory.unmapMemory();

    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);

    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
}

void vk_context::createIndexBuffer()
{
    vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    vk::raii::Buffer stagingBuffer({});
    vk::raii::DeviceMemory stagingBufferMemory({});
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void* data = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(data, indices.data(), bufferSize);
    stagingBufferMemory.unmapMemory();

    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

    copyBuffer(stagingBuffer, indexBuffer, bufferSize);
}

void vk_context::createUniformBuffers()
{
    uniformBuffers.clear();
    uniformBuffersMemory.clear();
    uniformBuffersMapped.clear();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
        vk::raii::Buffer buffer({});
        vk::raii::DeviceMemory bufferMem({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
        uniformBuffers.emplace_back(std::move(buffer));
        uniformBuffersMemory.emplace_back(std::move(bufferMem));
        uniformBuffersMapped.emplace_back( uniformBuffersMemory[i].mapMemory(0, bufferSize));
    }
}

void vk_context::createDescriptorPool()
{
    std::array poolSize {
        vk::DescriptorPoolSize( vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
        vk::DescriptorPoolSize(  vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT)
    };
    vk::DescriptorPoolCreateInfo poolInfo = {};
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSize.size());
    poolInfo.pPoolSizes = poolSize.data();

    descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
}

void vk_context::createDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo = {};
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.clear();
    descriptorSets = device.allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DescriptorBufferInfo bufferInfo = {};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        vk::DescriptorImageInfo imageInfo = {};
        imageInfo.sampler = textureSampler;
        imageInfo.imageView = textureImageView;
        imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

        vk::WriteDescriptorSet descriptorSet_1 = {};
        descriptorSet_1.dstSet = descriptorSets[i];
        descriptorSet_1.dstBinding = 0;
        descriptorSet_1.dstArrayElement = 0;
        descriptorSet_1.descriptorCount = 1;
        descriptorSet_1.descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorSet_1.pBufferInfo = &bufferInfo;
        
        vk::WriteDescriptorSet descriptorSet_2 = {};
        descriptorSet_2.dstSet = descriptorSets[i];
        descriptorSet_2.dstBinding = 1;
        descriptorSet_2.dstArrayElement = 0;
        descriptorSet_2.descriptorCount = 1;
        descriptorSet_2.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        descriptorSet_2.pImageInfo = &imageInfo;

        std::array descriptorWrites{
            descriptorSet_1,
            descriptorSet_2
        };
        device.updateDescriptorSets(descriptorWrites, {});
    }
}

void vk_context::createBuffer(vk::DeviceSize size,
                  vk::BufferUsageFlags usage,
                  vk::MemoryPropertyFlags properties,
                  vk::raii::Buffer& buffer,
                  vk::raii::DeviceMemory& bufferMemory)
{
    vk::BufferCreateInfo bufferInfo = {};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;

    buffer = vk::raii::Buffer(device, bufferInfo);
    vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo = {};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
    buffer.bindMemory(bufferMemory, 0);
}

std::unique_ptr<vk::raii::CommandBuffer> vk_context::beginSingleTimeCommands()
{
    vk::CommandBufferAllocateInfo allocInfo = {};
    allocInfo.commandPool = commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = 1;

    std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = std::make_unique<vk::raii::CommandBuffer>(std::move(vk::raii::CommandBuffers(device, allocInfo).front()));

    vk::CommandBufferBeginInfo beginInfo = {};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    commandBuffer->begin(beginInfo);

    return commandBuffer;
}

void vk_context::endSingleTimeCommands(const vk::raii::CommandBuffer& commandBuffer) const
{
    commandBuffer.end();

    vk::SubmitInfo submitInfo = {};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &*commandBuffer;

    queue.submit(submitInfo, nullptr);
    queue.waitIdle();
}

void vk_context::copyBuffer(vk::raii::Buffer & srcBuffer, vk::raii::Buffer & dstBuffer, vk::DeviceSize size)
{
    vk::CommandBufferAllocateInfo allocInfo = {};
    allocInfo.commandPool = commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = 1;

    vk::raii::CommandBuffer commandCopyBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());

    vk::CommandBufferBeginInfo cmdBufferBeginInfo = {};
    cmdBufferBeginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    /* NOTE: .begin() ask for a reference idk cpp shit, hay mas mierda igual .copyBuffer() y queue.submit() */
    commandCopyBuffer.begin(cmdBufferBeginInfo);
    /* 
     * Esta es la buena
     * vkBeginCommandBuffer(VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo *pBeginInfo);
    */

    vk::BufferCopy buffercpy = {};
    buffercpy.size = size;
    commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, buffercpy);

    commandCopyBuffer.end();

    vk::SubmitInfo submitInfo = {};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &*commandCopyBuffer;
    queue.submit(submitInfo, nullptr);
    queue.waitIdle();
}

uint32_t vk_context::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
{
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void vk_context::createCommandBuffers()
{
    commandBuffers.clear();
    vk::CommandBufferAllocateInfo allocInfo = {};
    allocInfo.commandPool = commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
    commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
}

void vk_context::recordCommandBuffer(uint32_t imageIndex)
{
    commandBuffers[currentFrame].begin({});
    // Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
    transition_image_layout(
        imageIndex,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        {},                                                     // srcAccessMask (no need to wait for previous operations)
        vk::AccessFlagBits2::eColorAttachmentWrite,                // dstAccessMask
        vk::PipelineStageFlagBits2::eTopOfPipe,                   // srcStage
        vk::PipelineStageFlagBits2::eColorAttachmentOutput        // dstStage
    );
    // Transition depth image to depth attachment optimal layout
    vk::ImageMemoryBarrier2 depthBarrier = {};
    depthBarrier.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
    depthBarrier.srcAccessMask = {};
    depthBarrier.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
    depthBarrier.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
    depthBarrier.oldLayout = vk::ImageLayout::eUndefined;
    depthBarrier.newLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    depthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthBarrier.image = depthImage;

    /* NOTE: Maybe it needs a fix bc using = is operator overloading cpp ¯\_(ツ)_/¯ */
    // depthBarrier.subresourceRange = {};
    depthBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
    depthBarrier.subresourceRange.baseMipLevel = 0;
    depthBarrier.subresourceRange.levelCount = 1;
    depthBarrier.subresourceRange.baseArrayLayer = 0;
    depthBarrier.subresourceRange.layerCount = 1;

    vk::DependencyInfo depthDependencyInfo = {};
    depthDependencyInfo.dependencyFlags = {};
    depthDependencyInfo.imageMemoryBarrierCount = 1;
    depthDependencyInfo.pImageMemoryBarriers = &depthBarrier;
    commandBuffers[currentFrame].pipelineBarrier2(depthDependencyInfo);

    vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

    /* NOTE: Maybe when rewritting in C, i can use {.member = value} and make the struct const, idk */
    vk::RenderingAttachmentInfo colorAttachmentInfo = {};
    colorAttachmentInfo.imageView = swapChainImageViews[imageIndex];
    colorAttachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    colorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachmentInfo.clearValue = clearColor;

    vk::RenderingAttachmentInfo depthAttachmentInfo = {};
    depthAttachmentInfo.imageView = depthImageView;
    depthAttachmentInfo.imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    depthAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
    depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachmentInfo.clearValue = clearDepth;

    vk::RenderingInfo renderingInfo = {};
    renderingInfo.renderArea = { .offset = { 0, 0 }, .extent = swapChainExtent };
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachmentInfo;
    renderingInfo.pDepthAttachment = &depthAttachmentInfo;

    commandBuffers[currentFrame].beginRendering(renderingInfo);
    commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
    commandBuffers[currentFrame].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffers[currentFrame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffers[currentFrame].bindVertexBuffers(0, *vertexBuffer, {0});
    commandBuffers[currentFrame].bindIndexBuffer( *indexBuffer, 0, vk::IndexType::eUint32 );
    commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[currentFrame], nullptr);
    commandBuffers[currentFrame].drawIndexed(indices.size(), 1, 0, 0, 0);
    commandBuffers[currentFrame].endRendering();

    // After rendering, transition the swapchain image to PRESENT_SRC
    transition_image_layout(
        imageIndex,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite,                 // srcAccessMask
        {},                                                      // dstAccessMask
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
        vk::PipelineStageFlagBits2::eBottomOfPipe                  // dstStage
    );
    commandBuffers[currentFrame].end();
}

void vk_context::transition_image_layout(
    uint32_t imageIndex,
    vk::ImageLayout old_layout,
    vk::ImageLayout new_layout,
    vk::AccessFlags2 src_access_mask,
    vk::AccessFlags2 dst_access_mask,
    vk::PipelineStageFlags2 src_stage_mask,
    vk::PipelineStageFlags2 dst_stage_mask
    )
{


    vk::ImageMemoryBarrier2 barrier = {};
    barrier.srcStageMask = src_stage_mask;
    barrier.srcAccessMask = src_access_mask;
    barrier.dstStageMask = dst_stage_mask;
    barrier.dstAccessMask = dst_access_mask;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = swapChainImages[imageIndex];
    
    /* NOTE: i dont think so but maybe need fix, it seems like doing that overloads = cpp ¯\_(ツ)_/¯ */
    // barrier.subresourceRange = {};
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    
    vk::DependencyInfo dependency_info = {};
    dependency_info.dependencyFlags = {};
    dependency_info.imageMemoryBarrierCount = 1;
    dependency_info.pImageMemoryBarriers = &barrier;

    commandBuffers[currentFrame].pipelineBarrier2(dependency_info);
}

void vk_context::createSyncObjects()
{
    presentCompleteSemaphore.clear();
    renderFinishedSemaphore.clear();
    inFlightFences.clear();

    for (size_t i = 0; i < swapChainImages.size(); i++) {
        presentCompleteSemaphore.emplace_back(device, vk::SemaphoreCreateInfo());
        renderFinishedSemaphore.emplace_back(device, vk::SemaphoreCreateInfo());
    }


    vk::FenceCreateInfo fenceInfo = {};
    fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        inFlightFences.emplace_back(device, fenceInfo);
    }
}

void vk_context::initVulkan()
{
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createCommandPool();
    createDepthResources();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    loadModel();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
}

void vk_context::updateUniformBuffer(uint32_t currentImage) const
{
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float>(currentTime - startTime).count();

    UniformBufferObject ubo{};
    ubo.model = rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.view = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f), static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 10.0f);
    ubo.proj[1][1] *= -1;

    memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

void vk_context::drawFrame()
{
    /* TODO: a while loop that does nothing, idk how it compiles, but i think taht i dont like it */
    while ( vk::Result::eTimeout == device.waitForFences( *inFlightFences[currentFrame], vk::True, UINT64_MAX ) )
        ;
    auto [result, imageIndex] = swapChain.acquireNextImage( UINT64_MAX, *presentCompleteSemaphore[semaphoreIndex], nullptr );

    if (result == vk::Result::eErrorOutOfDateKHR) {
        recreateSwapChain();
        return;
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }
    updateUniformBuffer(currentFrame);

    device.resetFences(  *inFlightFences[currentFrame] );
    commandBuffers[currentFrame].reset();
    recordCommandBuffer(imageIndex);

    vk::PipelineStageFlags waitDestinationStageMask( vk::PipelineStageFlagBits::eColorAttachmentOutput );
    
    /* TODO: Maybe irrelevant, but this was const */
    /* NOTE: Im quite sure that smthing like wath was here should work in cpp, idk, fuck this shit */
    vk::SubmitInfo submitInfo = {};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &*presentCompleteSemaphore[semaphoreIndex];
    submitInfo.pWaitDstStageMask = &waitDestinationStageMask;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &*commandBuffers[currentFrame];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &*renderFinishedSemaphore[imageIndex];

    queue.submit(submitInfo, *inFlightFences[currentFrame]);


    /* WFT is this */
    try {
        /* TODO: Maybe irrelevant, but this was const */
        /* NOTE: Im quite sure that smthing like wath was here should work in cpp, idk, fuck this shit */
        vk::PresentInfoKHR presentInfoKHR = {};
        presentInfoKHR.waitSemaphoreCount = 1;
        presentInfoKHR.pWaitSemaphores = &*renderFinishedSemaphore[imageIndex];
        presentInfoKHR.swapchainCount = 1;
        presentInfoKHR.pSwapchains = &*swapChain;
        presentInfoKHR.pImageIndices = &imageIndex;

        result = queue.presentKHR(presentInfoKHR);
        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }
    } catch (const vk::SystemError& e) {
        if (e.code().value() == static_cast<int>(vk::Result::eErrorOutOfDateKHR)) {
            recreateSwapChain();
            return;
        } else {
            throw;
        }
    }
    semaphoreIndex = (semaphoreIndex + 1) % presentCompleteSemaphore.size();
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

[[nodiscard]] vk::raii::ShaderModule vk_context::createShaderModule(const std::vector<char>& code) const
{
    vk::ShaderModuleCreateInfo createInfo = {};
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    vk::raii::ShaderModule shaderModule{device, createInfo};

    return shaderModule;
}

vk::Extent2D vk_context::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != 0xFFFFFFFF) {
        return capabilities.currentExtent;
    }
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        return {
            std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
        };
}

vk_context::~vk_context()
{
    LOG_FUNCTION()
}

void vk_context::run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

int main() {
    try {
        vk_context app;
        app.run();
        app.~vk_context();

        // Aqui segfault, el destructor ya se ha llamado
        // Creo que puede ser por lo de que dynamicRendering no se ha activado
        // EL codigo del tutorial sin modificar tambien segfault
        // aunque sin quejas de validationLayers
        // Parece que los compila para c++11 sin -g
        // las validation layers estan activas

        std::cout << "Hola" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "I hate cpp" << std::endl;

    return EXIT_SUCCESS;
}
