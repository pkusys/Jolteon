#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
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
#include <aws/lambda-runtime/runtime.h>
#include <iostream>
#include <memory>

#include <aws/lambda-runtime/runtime.h>

#include "csortlib.h"

using namespace csortlib;

using namespace aws::lambda_runtime;

void PrintRecord(const Record &rec) {
  for (size_t i = 0; i < HEADER_SIZE; ++i) {
    printf("%02x ", rec.header[i]);
  }
  printf("\n");
}

void AssertSorted(const Array<Record> &array) {
  for (size_t i = 0; i < array.size - 1; ++i) {
    const auto &a = array.ptr[i];
    const auto &b = array.ptr[i + 1];
    assert(std::memcmp(a.header, b.header, HEADER_SIZE) <= 0);
  }
}

std::vector<ConstArray<Record>>
MakeConstRecordArrays(Record *const records,
                      const std::vector<Partition> &parts) {
  std::vector<ConstArray<Record>> ret;
  ret.reserve(parts.size());
  for (const auto &part : parts) {
    ret.emplace_back(ConstArray<Record>{records + part.offset, part.size});
  }
  return ret;
}


std::string download_file(
    Aws::S3::S3Client const& client,
    Aws::String const& bucket,
    Aws::String const& key,
    Aws::String& output);

std::string upload_file(
    Aws::S3::S3Client const& client,
    Aws::String const& bucket,
    Aws::String const& key,
    Aws::String const& body);

std::string read_to_string(Aws::IOStream& stream, Aws::String& output);
char const TAG[] = "LAMBDA_ALLOC";

invocation_response my_handler(invocation_request const& req, Aws::S3::S3Client const& client)
{

  using namespace Aws::Utils::Json;
  JsonValue json(req.payload);
  if (!json.WasParseSuccessful()) {
      return invocation_response::failure("Failed to parse input JSON", "InvalidJSON");
  }

  auto v = json.View();

  if (!v.ValueExists("s3bucket") || !v.ValueExists("s3key") || !v.GetObject("s3bucket").IsString() ||
      !v.GetObject("s3key").IsString()) {
      return invocation_response::failure("Missing input value s3bucket or s3key", "InvalidJSON");
  }

  auto bucket = v.GetString("s3bucket");
  auto key = v.GetString("s3key");

  AWS_LOGSTREAM_INFO(TAG, "Attempting to download file from s3://" << bucket << "/" << key);

  Aws::String rec_string;
  auto err = download_file(client, bucket, key, rec_string);
  if (!err.empty()) {
      return invocation_response::failure(err, "DownloadFailure");
  }

  const size_t num_reducers = 10;
  const auto &boundaries = GetBoundaries(num_reducers);

  std::vector<std::string> files;

  // Read number from files, s3 read
  size_t num_records = rec_string.size() / HEADER_SIZE;

  Record *records = new Record[num_records];

  for (int i = 0; i < num_records; i++){
    uint8_t *p = (uint8_t *)(records + i);
    memcpy(p, rec_string.c_str() + i * HEADER_SIZE, HEADER_SIZE);
  }

  auto ret = SortAndPartition({records, num_records}, boundaries);

  // for (size_t i = 0; i < ret.size(); i++){
  //   printf("%ld %ld\n", ret[i].offset, ret[i].size);
  // }



  const auto& record_arrays = MakeConstRecordArrays(records, ret);
  const auto output = MergePartitions(record_arrays);

  // FILE* fout;
  // fout = fopen("data1g-output", "w");
  // if (fout == NULL) {
  //     perror("Failed to open file");
  // } else {
  //     size_t writecount = fwrite(output.ptr, RECORD_SIZE, output.size, fout);
  //     printf("Wrote %lu bytes.\n", writecount);
  //     fclose(fout);
  // }


  return invocation_response::success("Hello, World!", "application/json");
}

std::function<std::shared_ptr<Aws::Utils::Logging::LogSystemInterface>()> GetConsoleLoggerFactory()
{
    return [] {
        return Aws::MakeShared<Aws::Utils::Logging::ConsoleLogSystem>(
            "console_logger", Aws::Utils::Logging::LogLevel::Trace);
    };
}

std::string upload_file(
    Aws::S3::S3Client const& client,
    Aws::String const& bucket,
    Aws::String const& key,
    Aws::String const& body)
{
    using namespace Aws;

    Aws::S3::Model::PutObjectRequest request;
    request.SetBucket(bucket);
    request.SetKey(key);
    std::shared_ptr<Aws::IOStream> bodyStream = 
        Aws::MakeShared<Aws::StringStream>("PutObjectInputStream");
    *bodyStream << body;
    request.SetBody(bodyStream);

    auto outcome = client.PutObject(request);
    if (outcome.IsSuccess()) {
        AWS_LOGSTREAM_INFO(TAG, "Upload completed!");
        return {};
    }
    else {
        AWS_LOGSTREAM_ERROR(TAG, "Failed with error: " << outcome.GetError());
        return outcome.GetError().GetMessage();
    }
}

std::string read_to_string(Aws::IOStream& stream, Aws::String& output)
{
    Aws::Vector<unsigned char> bits;
    bits.reserve(stream.tellp());
    stream.seekg(0, stream.beg);

    char streamBuffer[1024 * 4];
    while (stream.good()) {
        stream.read(streamBuffer, sizeof(streamBuffer));
        auto bytesRead = stream.gcount();

        if (bytesRead > 0) {
            bits.insert(bits.end(), (unsigned char*)streamBuffer, (unsigned char*)streamBuffer + bytesRead);
        }
    }
    // Aws::Utils::ByteBuffer bb(bits.data(), bits.size());
    // output = Aws::Utils::HashingUtils::Base64Encode(bb);
    output = Aws::String((char *)bits.data(), bits.size());
    return {};
}

std::string write_to_binary(Aws::IOStream& stream, Aws::String& input) {
    stream.write(input.c_str(), input.size());
    return {};
}

std::string download_file(
    Aws::S3::S3Client const& client,
    Aws::String const& bucket,
    Aws::String const& key,
    Aws::String& output)
{
    using namespace Aws;

    S3::Model::GetObjectRequest request;
    request.WithBucket(bucket).WithKey(key);

    auto outcome = client.GetObject(request);
    if (outcome.IsSuccess()) {
        AWS_LOGSTREAM_INFO(TAG, "Download completed!");
        auto& s = outcome.GetResult().GetBody();
        return read_to_string(s, output);
    }
    else {
        AWS_LOGSTREAM_ERROR(TAG, "Failed with error: " << outcome.GetError());
        return outcome.GetError().GetMessage();
    }
}

int main(int argc, char* argv[]) {

  using namespace Aws;
  SDKOptions options;
  options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Info;
//   options.loggingOptions.logger_create_fn = GetConsoleLoggerFactory();
  InitAPI(options);
  {
    // Client::ClientConfiguration config;
    // config.region = Aws::Environment::GetEnv("AWS_REGION");
    // config.caFile = "/etc/pki/tls/certs/ca-bundle.crt";

    // auto credentialsProvider = Aws::MakeShared<Aws::Auth::EnvironmentAWSCredentialsProvider>(TAG);
    // S3::S3Client client(credentialsProvider, config);
    S3::S3Client client;

    // auto handler_fn = [&client](aws::lambda_runtime::invocation_request const& req) {
    //     return my_handler(req, client);
    // };
    //run_handler(handler_fn);

    std::ifstream fin;
	fin.open("../part1", std::ios::in | std::ios::binary);
	Aws::String p1;
	fin.seekg(0, std::ios::end);
	p1.reserve(fin.tellg());
	fin.seekg(0, std::ios::beg);
	p1.assign((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
	fin.close();
	auto err0 = upload_file(client, "serverless-bound", "p1", p1);

	Aws::String part1;
	auto err = download_file(client, "serverless-bound", "p1", part1);
	// write to local file
	std::ofstream fout;
	fout.open("part1-download", std::ios::out | std::ios::binary);
	fout << part1;
	fout.close();
  }
  ShutdownAPI(options);
  return 0;
}
