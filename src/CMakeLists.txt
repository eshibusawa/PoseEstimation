cmake_minimum_required(VERSION 3.3)

project(PoseEstimation)

message(" C++ module")
if (UNIX)
	find_package(PkgConfig)
	pkg_search_module(CppUnit cppunit)
	set(CppUnit_LIBRARY_DIRS ${CppUnit_LIBDIR})
	set(CppUnit_LIBRARIES ${CppUnit_LIBRARIES} CACHE STRING "")

	set(CPPUNIT_FOUND "")
	if (CppUnit_LIBRARIES)
		message("   CppUnit enabled")
		set(CPPUNIT_FOUND TRUE)
	endif ()

	find_package(Eigen3 REQUIRED eigen3)
	set(EIGEN3_FOUND "")
	if (EIGEN3_INCLUDE_DIR)
		message("   Eigen3 enabled")
	endif ()
endif ()

set(CMAKE_CXX_STANDARD 11)
if (MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else ()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
	set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
	set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
endif()
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

if (CPPUNIT_FOUND)
	enable_testing()
	add_executable(PoseEstimationTest
		UnitTestMain.cpp
		FivePointTest.cpp
		SevenPointTest.cpp
		ThreePointTest.cpp
		P3PTest.cpp
		)
	target_include_directories(PoseEstimationTest PRIVATE ${CppUnit_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ./)
	target_link_libraries(PoseEstimationTest ${CppUnit_LIBRARIES})
	install(TARGETS PoseEstimationTest DESTINATION .)
	add_test(NAME PoseEstimationTest COMMAND PoseEstimationTest)
endif ()

add_executable(EPnPMain
	EPnPMain.cpp
	PoseUtil.cpp
	)
target_include_directories(EPnPMain PRIVATE ${EIGEN3_INCLUDE_DIR} ./)
target_link_libraries(EPnPMain )
install(TARGETS EPnPMain DESTINATION .)

add_executable(P3PMain
	P3PMain.cpp
	)
target_include_directories(P3PMain PRIVATE ${EIGEN3_INCLUDE_DIR} ./)
target_link_libraries(P3PMain )
install(TARGETS P3PMain DESTINATION .)

add_executable(FivePointMain
	FivePointMain.cpp
	)
target_include_directories(FivePointMain PRIVATE ${EIGEN3_INCLUDE_DIR} ./)
target_link_libraries(FivePointMain )
install(TARGETS FivePointMain DESTINATION .)

add_executable(ThreePointMain
	ThreePointMain.cpp
	)
target_include_directories(ThreePointMain PRIVATE ${EIGEN3_INCLUDE_DIR} ./)
target_link_libraries(ThreePointMain )
install(TARGETS ThreePointMain DESTINATION .)
