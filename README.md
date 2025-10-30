i hate cpp, but i want to continue with the tutorial, using cpp make that easier
Lo de los tiempos de compilacion es una verguenza XD

Hay distinto comportamiento en x11 y wayland, en ambos segfault
x11 no funiona el resize, en el debugger si funciona XD
"""
02:39:06 [1] xiri@glados ~/repos/vulkan_tuto_cpp $ XDG_SESSION_TYPE=x11 ./vulkan
Function ~vk_context() called in src/main.cpp at line 1453
vk::raii::SwapchainKHR::acquireNextImage: ErrorOutOfDateKHR
02:39:16 [1] xiri@glados ~/repos/vulkan_tuto_cpp $ XDG_SESSION_TYPE=x11 ./vulkan
Function cleanup() called in src/main.cpp at line 377
Function ~vk_context() called in src/main.cpp at line 1453
Hola
Function ~vk_context() called in src/main.cpp at line 1453
Segmentation fault         (core dumped) XDG_SESSION_TYPE=x11 ./vulkan
02:40:17 [139] xiri@glados ~/repos/vulkan_tuto_cpp $ XDG_SESSION_TYPE=wayland ./vulkan
Function cleanup() called in src/main.cpp at line 377
Function ~vk_context() called in src/main.cpp at line 1453
Segmentation fault         (core dumped) XDG_SESSION_TYPE=wayland ./vulkan
"""
