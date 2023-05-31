#include <aws/lambda-runtime/runtime.h>
#include <chrono>

#include "csortlib.h"
#include "io.h"

using namespace csortlib;

using namespace aws::lambda_runtime;

/* Get current time */
std::chrono::high_resolution_clock::time_point get_time() {
    return std::chrono::high_resolution_clock::now();
}

/* Get time duration in milliseconds */
long get_time_duration_ms(std::chrono::high_resolution_clock::time_point start, 
                       std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

/* Get time duration in microseconds */
long get_time_duration_us(std::chrono::high_resolution_clock::time_point start, 
                       std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

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

std::vector<ConstArray<Record> > 
MakeConstRecordArrays(Record *const records,
                      const std::vector<Partition> &parts) {
    std::vector<ConstArray<Record>> ret;
    ret.reserve(parts.size());
    for (const auto &part : parts) {
        ret.emplace_back(ConstArray<Record>{records + part.offset, part.size});
    }
    return ret;
}

char const TAG[] = "LAMBDA_ALLOC";

invocation_response my_handler(invocation_request const& req, Aws::S3::S3Client const& client) {
    using namespace Aws::Utils::Json;

    // Parse input JSON
    auto start_time = get_time();
    JsonValue json(req.payload);
    if (!json.WasParseSuccessful()) {
        return invocation_response::failure("Failed to parse input JSON", "InvalidJSON");
    }
    auto v = json.View();

    // Get s3 info
    // input data bucket/key
    if (!v.ValueExists("s3bucket_in") || !v.ValueExists("s3key_in") || 
        !v.GetObject("s3bucket_in").IsString() || !v.GetObject("s3key_in").IsString()) {
        return invocation_response::failure("Missing input value s3bucket_in or s3key_in", 
                                            "InvalidJSON");
    }
    auto bucket_in = v.GetString("s3bucket_in");
    auto key_in = v.GetString("s3key_in");
    // output data bucket/key
    if (!v.ValueExists("s3bucket_out") || !v.ValueExists("s3key_out") || 
        !v.GetObject("s3bucket_out").IsString() || !v.GetObject("s3key_out").IsString()) {
        return invocation_response::failure("Missing input value s3bucket_out or s3key_out", 
                                            "InvalidJSON");
    }
    auto bucket_out = v.GetString("s3bucket_out");
    auto key_out = v.GetString("s3key_out");

    // Get func_type
    if (!v.ValueExists("func_type") || !v.GetObject("func_type").IsString()) {
        return invocation_response::failure("Missing input value func_type", 
                                            "InvalidJSON");
    }
    auto func_type = v.GetString("func_type");
    if (func_type != "map" && func_type != "reduce") {
        return invocation_response::failure("Invalid func_type", "InvalidJSON");
    }

    // Get number of map tasks and number of reduce tasks
    if (!v.ValueExists("num_mappers") || !v.ValueExists("num_reducers") ||
        !v.GetObject("num_mappers").IsIntegerType() || 
        !v.GetObject("num_reducers").IsIntegerType()) {
        return invocation_response::failure("Missing input value num_mappers or num_reducers", 
                                            "InvalidJSON");
    }
    const size_t num_mappers = v.GetInteger("num_mappers");
    const size_t num_reducers = v.GetInteger("num_reducers");

    // Get task id and number of partitions of raw input data
    if (!v.ValueExists("task_id") || !v.ValueExists("num_partitions") ||
        !v.GetObject("task_id").IsIntegerType() || 
        !v.GetObject("num_partitions").IsIntegerType()) {
        return invocation_response::failure("Missing input value task_id or num_partitions", 
                                            "InvalidJSON");
    }
    const size_t task_id = v.GetInteger("task_id");
    const size_t num_partitions = v.GetInteger("num_partitions");

    if (num_mappers == 0 || num_reducers == 0 || num_partitions == 0) {
        return invocation_response::failure(
            "Zero value of num_mappers or num_reducers or num_partitions", "InvalidJSON");
    }

    auto end_time = get_time();
    auto parse_duration = get_time_duration_ms(start_time, end_time);

    long read_duration = 0;
    long records_creation_duration = 0;
    long sort_duration = 0;
    long write_duration = 0;

    if (func_type == "map") {
        start_time = get_time();
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
            Aws::String part_key = key_in + "_" + std::to_string(i);
            auto err = download_file_binary(client, bucket_in, part_key, record_bytes);
            if (!err.empty()) {
                return invocation_response::failure(err, "DownloadFailure");
            }
        }
        end_time = get_time();
        read_duration = get_time_duration_ms(start_time, end_time);

        // Create records from record_bytes
        start_time = get_time();
        size_t num_records = record_bytes.size() / RECORD_SIZE;
        Record *records = new Record[num_records];
        memcpy((uint8_t *)records, record_bytes.data(), num_records * RECORD_SIZE);
        // Free memory
        {
            ByteVec().swap(record_bytes);
        }
        end_time = get_time();
        records_creation_duration = get_time_duration_ms(start_time, end_time);

        // Sort and partition
        start_time = get_time();

        const auto &boundaries = GetBoundaries(num_reducers);
        auto ret = SortAndPartition({records, num_records}, boundaries);

        end_time = get_time();
        sort_duration = get_time_duration_ms(start_time, end_time);

        // Upload partitions to s3
        start_time = get_time();
        const auto &record_arrays = MakeConstRecordArrays(records, ret);
        for (size_t i = 0; i < record_arrays.size(); ++i) {
            const auto &partition = record_arrays[i];
            Aws::String part_key = key_out + "_map" + std::to_string(task_id) + 
                                   "_part" + std::to_string(i);
            ByteVec part_data; 
            // Convert partition to string
            ConvertRecordArrayToBinary(partition, part_data);
            auto err = upload_file_binary(client, bucket_out, part_key, part_data); 
            if (!err.empty()) {
                return invocation_response::failure(err, "UploadFailure");
            }
        }
        end_time = get_time();
        write_duration = get_time_duration_ms(start_time, end_time);

        delete[] records;
    } else {
        start_time = get_time();
        // Download all partitions belonging to this reducer from s3
        ByteVec record_bytes;
        std::vector<Partition> partitions;
        partitions.reserve(num_mappers);
        
        size_t num_bytes = 0;
        for (size_t i = 0; i < num_mappers; i++) {
            Aws::String part_key = key_in + "_map" + std::to_string(i) + 
                                   "_part" + std::to_string(task_id);
            auto err = download_file_binary(client, bucket_in, part_key, record_bytes);
            if (!err.empty()) {
                return invocation_response::failure(err, "DownloadFailure");
            }

            auto num_records = (record_bytes.size() - num_bytes) / RECORD_SIZE;
            auto offset = num_bytes / RECORD_SIZE;
            num_bytes = record_bytes.size();
            partitions.emplace_back(Partition{offset, num_records});
        }
        end_time = get_time();
        read_duration = get_time_duration_ms(start_time, end_time);

        // Create records from record_bytes
        start_time = get_time();
        Record *records = new Record[num_bytes / RECORD_SIZE];
        memcpy((uint8_t *)records, record_bytes.data(), num_bytes);
        // Free memory
        {
            ByteVec().swap(record_bytes);
        }
        end_time = get_time();
        records_creation_duration = get_time_duration_ms(start_time, end_time);

        // Merge partitions with sort
        start_time = get_time();

        const auto record_arrays = MakeConstRecordArrays(records, partitions);
        const auto result = MergePartitions(record_arrays);

        end_time = get_time();
        sort_duration = get_time_duration_ms(start_time, end_time);

        // Upload the result to s3
        start_time = get_time();
        Aws::String res_key = key_out + "_reduce" + std::to_string(task_id);
        ByteVec result_bytes;
        ConvertRecordArrayToBinary(result, result_bytes);
        auto err = upload_file_binary(client, bucket_out, res_key, result_bytes);
        if (!err.empty()) {
            return invocation_response::failure(err, "UploadFailure");
        }
        end_time = get_time();
        write_duration = get_time_duration_ms(start_time, end_time);

        delete[] records;
    }

    return invocation_response::success("Terasort " + func_type + " task,"
                                        " id " + std::to_string(task_id) + " ,"
                                        " num_mappers " + std::to_string(num_mappers) + " ," + 
                                        " num_reducers " + std::to_string(num_reducers) + " /" +
                                        " parse_duration " + std::to_string(parse_duration) + " ms," + 
                                        " read_duration " + std::to_string(read_duration) + " ms," + 
                                        " record_creation_duration " + std::to_string(records_creation_duration) + " ms," + 
                                        " sort_duration " + std::to_string(sort_duration) + " ms," + 
                                        " write_duration " + std::to_string(write_duration) + " ms", 
                                        "application/json");
}

int main(int argc, char* argv[]) {
    using namespace Aws;

    SDKOptions options;
    options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Info;
    // Do not ignore SIGPIPE in AWS SDK C++
    // Refer to https://repost.aws/questions/QUq-v7qLPmTli8ExAZ-16Iaw/cognito-c-sdk-broken-pipe
    options.httpOptions.installSigPipeHandler = true;
    // options.loggingOptions.logger_create_fn = GetConsoleLoggerFactory();
    InitAPI(options);
    {
        Client::ClientConfiguration config;
        config.region = "us-east-1";
        config.caFile = "/etc/pki/tls/certs/ca-bundle.crt";

        // auto credentialsProvider = 
        //     Aws::MakeShared<Aws::Auth::EnvironmentAWSCredentialsProvider>(TAG);
        // S3::S3Client client(credentialsProvider, config);
        S3::S3Client client(config);

        auto handler_fn = [&client](aws::lambda_runtime::invocation_request const& req) {
            return my_handler(req, client);
        };
        run_handler(handler_fn);

        // test_s3io_bin(client, "serverless-bound", "terasort/test/test-32m_0");
    }
    ShutdownAPI(options);
    return 0;
}
