cmake_minimum_required(VERSION 3.3)

project(PoseEstimation)

message(" C++ module")
if (UNIX)
	find_package(PkgConfig)
	find_package(Eigen3 REQUIRED eigen3)
	set(EIGEN3_FOUND "")
	if (EIGEN3_INCLUDE_DIR)
		message("   Eigen3 enabled")
	endif ()
endif ()

if (MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else ()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
	set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
	set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
endif()

add_executable(EPnPMain
	EPnPMain.cpp
	PoseUtil.cpp
	)
target_include_directories(EPnPMain PRIVATE ${EIGEN3_INCLUDE_DIR} ./)
target_link_libraries(EPnPMain )
install(TARGETS EPnPMain DESTINATION .)
