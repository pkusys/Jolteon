cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 11)
project(main LANGUAGES CXX)

add_library(sortlib csortlib.cpp)

find_package(aws-lambda-runtime REQUIRED)
find_package(AWSSDK COMPONENTS s3)
add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} PUBLIC
                      AWS::aws-lambda-runtime
                       ${AWSSDK_LINK_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC sortlib)

aws_lambda_package_target(${PROJECT_NAME})