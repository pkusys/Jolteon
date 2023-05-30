#include <aws/lambda-runtime/runtime.h>

#include "csortlib.h"
#include "io.h"

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


// std::string read_to_string(Aws::IOStream& stream, Aws::String& output);
char const TAG[] = "LAMBDA_ALLOC";

invocation_response my_handler(invocation_request const& req, Aws::S3::S3Client const& client)
{

  using namespace Aws::Utils::Json;
  JsonValue json(req.payload);
  if (!json.WasParseSuccessful()) {
      return invocation_response::failure("Failed to parse input JSON", "InvalidJSON");
  }

  auto v = json.View();

  // Get s3 info
  // input data bucket/key
  if (!v.ValueExists("s3bucket_in") || !v.ValueExists("s3key_in") || !v.GetObject("s3bucket_in").IsString() ||
      !v.GetObject("s3key_in").IsString()) {
      return invocation_response::failure("Missing input value s3bucket_in or s3key_in", "InvalidJSON");
  }
  auto bucket_in = v.GetString("s3bucket_in");
  auto key_in = v.GetString("s3key_in");
  // output data bucket/key
  if (!v.ValueExists("s3bucket_out") || !v.ValueExists("s3key_out") || !v.GetObject("s3bucket_out").IsString() ||
      !v.GetObject("s3key_out").IsString()) {
      return invocation_response::failure("Missing input value s3bucket_out or s3key_out", "InvalidJSON");
  }
  auto bucket_out = v.GetString("s3bucket_out");
  auto key_out = v.GetString("s3key_out");

  // Get func_type
  if (!v.ValueExists("func_type") || !v.GetObject("func_type").IsString()) {
      return invocation_response::failure("Missing input value func_type", "InvalidJSON");
  }
  auto func_type = v.GetString("func_type");
  if (func_type != "map" && func_type != "reduce") {
      return invocation_response::failure("Invalid func_type", "InvalidJSON");
  }

  // Get number of map tasks and number of reduce tasks
  if (!v.ValueExists("num_mappers") || !v.ValueExists("num_reducers") ||
      !v.GetObject("num_mappers").IsIntegerType() || !v.GetObject("num_reducers").IsIntegerType()) {
      return invocation_response::failure("Missing input value num_mappers or num_reducers", "InvalidJSON");
  }
  const size_t num_mappers = v.GetInteger("num_map");
  const size_t num_reducers = v.GetInteger("num_reduce");

  // Get task id and number of partitions of raw input data
  if (!v.ValueExists("task_id") || !v.ValueExists("num_partitions") ||
      !v.GetObject("task_id").IsIntegerType() || !v.GetObject("num_partitions").IsIntegerType()) {
      return invocation_response::failure("Missing input value task_id or num_partitions", "InvalidJSON");
  }
  const size_t task_id = v.GetInteger("task_id");
  const size_t num_partitions = v.GetInteger("num_partitions");

  if (num_mappers == 0 || num_reducers == 0 || num_partitions == 0) {
      return invocation_response::failure("Zero value of num_mappers or num_reducers or num_partitions", "InvalidJSON");
  }

  if (func_type == "map") {
      // Compute which partitions to process
      size_t parts_per_task = num_partitions / num_mappers;
      size_t remainder = num_partitions % num_mappers;
      size_t start = 0, end = 0;
      if (task_id < remainder) {
          start = task_id * (parts_per_task + 1);
          end = start + parts_per_task + 1;
      } else {
          start = remainder * (parts_per_task + 1) + (task_id - remainder) * parts_per_task;
          end = start + parts_per_task;
      }

      // Download records from s3, all append to record_bytes
      ByteVec record_bytes;
      for (size_t i = start; i < end; i++) {
          Aws::String part_key = key_in + "_" + std::to_string(start + i);
          auto err = download_file_binary(client, bucket_in, part_key, record_bytes);
          if (!err.empty()) {
              return invocation_response::failure(err, "DownloadFailure");
          }
      }

      // Sort and partition
      const auto &boundaries = GetBoundaries(num_reducers);
      
      size_t num_records = record_bytes.size() / HEADER_SIZE;
      Record *records = new Record[num_records];
      for (size_t i = 0; i < num_records; i++){
          uint8_t *p = (uint8_t *)(records + i);
          memcpy(p, record_bytes.data() + i * HEADER_SIZE, HEADER_SIZE);
      }

      auto ret = SortAndPartition({records, num_records}, boundaries);
      // for (size_t i = 0; i < ret.size(); i++){
      //   printf("%ld %ld\n", ret[i].offset, ret[i].size);
      // }

      const auto &record_arrays = MakeConstRecordArrays(records, ret);
      for (size_t i = 0; i < record_arrays.size(); ++i) {
          const auto &partition = record_arrays[i];
          Aws::String part_key = key_out + "_map" + std::to_string(task_id) + "_part" + std::to_string(i);
          ByteVec part_data; 
          // Convert partition to string
          ConvertRecordArrayToBinary(partition, part_data);
          auto err = upload_file_binary(client, bucket_out, part_key, part_data); 
          if (!err.empty()) {
              return invocation_response::failure(err, "UploadFailure");
        }
      }
  } else {
      // TODO: read partitions from s3
      // const auto& record_arrays = MakeConstRecordArrays(records, ret);
      // const auto output = MergePartitions(record_arrays);

      // FILE* fout;
      // fout = fopen("data1g-output", "w");
      // if (fout == NULL) {
      //     perror("Failed to open file");
      // } else {
      //     size_t writecount = fwrite(output.ptr, RECORD_SIZE, output.size, fout);
      //     printf("Wrote %lu bytes.\n", writecount);
      //     fclose(fout);
      // }
  }

  return invocation_response::success("Hello, World!", "application/json");
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

    auto handler_fn = [&client](aws::lambda_runtime::invocation_request const& req) {
        return my_handler(req, client);
    };
    run_handler(handler_fn);

    // test_s3_io(client, "serverless-bound", "p1");
  }
  ShutdownAPI(options);
  return 0;
}
