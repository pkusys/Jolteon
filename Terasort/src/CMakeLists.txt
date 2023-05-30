cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 11)
project(main LANGUAGES CXX)

add_library(sortlib csortlib.cpp)

find_package(aws-lambda-runtime REQUIRED)
find_package(AWSSDK COMPONENTS s3)
find_package(ZLIB)

add_library(iolib io.cpp)
target_link_libraries(iolib PUBLIC ${AWSSDK_LINK_LIBRARIES})

add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} PUBLIC
                      AWS::aws-lambda-runtime
                       ${AWSSDK_LINK_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC sortlib)
target_link_libraries(${PROJECT_NAME} PUBLIC iolib)

aws_lambda_package_target(${PROJECT_NAME})