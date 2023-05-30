#ifndef __IO_H__
#define __IO_H__

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <cassert>
#include <chrono>

#include <aws/core/Aws.h>
#include <aws/core/utils/logging/LogLevel.h>
#include <aws/core/utils/logging/ConsoleLogSystem.h>
#include <aws/core/utils/logging/LogMacros.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/platform/Environment.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/PutObjectRequest.h>

#include "csortlib.h"

typedef std::vector<unsigned char> ByteVec;

std::function<std::shared_ptr<Aws::Utils::Logging::LogSystemInterface>()> GetConsoleLoggerFactory();

std::string download_file_string(
    Aws::S3::S3Client const& client,
    Aws::String const& bucket,
    Aws::String const& key,
    Aws::String& output);

std::string upload_file_string(
    Aws::S3::S3Client const& client,
    Aws::String const& bucket,
    Aws::String const& key,
    Aws::String const& body);

std::string download_file_binary(
    Aws::S3::S3Client const& client,
    Aws::String const& bucket,
    Aws::String const& key,
    ByteVec& output);

std::string upload_file_binary(
    Aws::S3::S3Client const& client,
    Aws::String const& bucket,
    Aws::String const& key,
    ByteVec const& body);

/* Convert a sorted record array to an Aws::String */
// void ConvertRecordArrayToString(
//     const csortlib::ConstArray<Record> &record_array, Aws::String &output);

/* Convert a sorted record array to a std::vector<unsigned char> */
void ConvertRecordArrayToBinary(const csortlib::ConstArray<csortlib::Record> &record_array, ByteVec &output);

void test_s3_io(Aws::S3::S3Client const& client,
    Aws::String const& bucket,
    Aws::String const& key);

#endif
