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

if (MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else ()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
	set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
	set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
endif()

if (CPPUNIT_FOUND)
	add_executable(FivePointTest
		UnitTestMain.cpp
		FivePointTest.cpp
		)
	target_include_directories(FivePointTest PRIVATE ${CppUnit_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ./)
	target_link_libraries(FivePointTest ${CppUnit_LIBRARIES})
	install(TARGETS FivePointTest DESTINATION .)
endif ()

add_executable(EPnPMain
	EPnPMain.cpp
	PoseUtil.cpp
	)
target_include_directories(EPnPMain PRIVATE ${EIGEN3_INCLUDE_DIR} ./)
target_link_libraries(EPnPMain )
install(TARGETS EPnPMain DESTINATION .)

add_executable(FivePointMain
	FivePointMain.cpp
	)
target_include_directories(FivePointMain PRIVATE ${EIGEN3_INCLUDE_DIR} ./)
target_link_libraries(FivePointMain )
install(TARGETS FivePointMain DESTINATION .)
