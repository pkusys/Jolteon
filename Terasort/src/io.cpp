#include "io.h"

std::function<std::shared_ptr<Aws::Utils::Logging::LogSystemInterface>()> 
GetConsoleLoggerFactory() {
    return [] {
        return Aws::MakeShared<Aws::Utils::Logging::ConsoleLogSystem>(
            "console_logger", Aws::Utils::Logging::LogLevel::Trace);
    };
}

std::string upload_file_string(
    Aws::S3::S3Client const& client,
    Aws::String const& bucket,
    Aws::String const& key,
    Aws::String const& body) {
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
        //AWS_LOGSTREAM_INFO(TAG, "Upload completed!");
        return {};
    }
    else {
        //AWS_LOGSTREAM_ERROR(TAG, "Failed with error: " << outcome.GetError());
        return outcome.GetError().GetMessage();
    }
}

std::string read_to_string(Aws::IOStream& stream, Aws::String& output) {
    Aws::Vector<unsigned char> bytes;
    bytes.reserve(bytes.size() + stream.tellp());
    stream.seekg(0, stream.beg);

    char streamBuffer[1024 * 4];
    while (stream.good()) {
        stream.read(streamBuffer, sizeof(streamBuffer));
        auto bytesRead = stream.gcount();

        if (bytesRead > 0) {
            bytes.insert(bytes.end(), (unsigned char*)streamBuffer, (unsigned char*)streamBuffer + bytesRead);
        }
    }
    // Aws::Utils::ByteBuffer bb(bytes.data(), bytes.size());
    // output = Aws::Utils::HashingUtils::Base64Encode(bb);
    // output = Aws::String((char *)bytes.data(), bytes.size());

    // append to output
    output += Aws::String((char *)bytes.data(), bytes.size());
    return {};
}

std::string download_file_string(
    Aws::S3::S3Client const& client,
    Aws::String const& bucket,
    Aws::String const& key,
    Aws::String& output) {
    using namespace Aws;

    S3::Model::GetObjectRequest request;
    request.WithBucket(bucket).WithKey(key);

    auto outcome = client.GetObject(request);
    if (outcome.IsSuccess()) {
        //AWS_LOGSTREAM_INFO(TAG, "Download completed!");
        auto& s = outcome.GetResult().GetBody();
        return read_to_string(s, output);
    }
    else {
        //AWS_LOGSTREAM_ERROR(TAG, "Failed with error: " << outcome.GetError());
        return outcome.GetError().GetMessage();
    }
}

std::string upload_file_binary(
    Aws::S3::S3Client const& client,
    Aws::String const& bucket,
    Aws::String const& key,
    ByteVec const& body) {
    using namespace Aws;

    Aws::S3::Model::PutObjectRequest request;
    request.SetBucket(bucket);
    request.SetKey(key);
    std::shared_ptr<Aws::IOStream> bodyStream = 
        Aws::MakeShared<Aws::StringStream>("PutObjectInputStream");
    bodyStream->write((char *)body.data(), body.size());
    request.SetBody(bodyStream);

    auto outcome = client.PutObject(request);
    if (outcome.IsSuccess()) {
        return {};
    }
    else {
        return outcome.GetError().GetMessage();
    }
}

std::string read_to_binary(Aws::IOStream& stream, ByteVec& output)
{
    // append to output
    char streamBuffer[1024 * 4];
    output.reserve(output.size() + stream.tellp());
    stream.seekg(0, stream.beg);

    while (stream.good()) {
        stream.read(streamBuffer, sizeof(streamBuffer));
        auto bytesRead = stream.gcount();

        if (bytesRead > 0) {
            output.insert(output.end(), (unsigned char*)streamBuffer, 
                          (unsigned char*)streamBuffer + bytesRead);
        }
    }
    return {};
}

std::string download_file_binary(
    Aws::S3::S3Client const& client,
    Aws::String const& bucket,
    Aws::String const& key,
    ByteVec& output) {
    using namespace Aws;

    S3::Model::GetObjectRequest request;
    request.WithBucket(bucket).WithKey(key);

    auto outcome = client.GetObject(request);
    if (outcome.IsSuccess()) {
        auto& s = outcome.GetResult().GetBody();
        return read_to_binary(s, output);
    }
    else {
        return outcome.GetError().GetMessage();
    }
}

void ConvertRecordArrayToBinary(
    const csortlib::ConstArray<csortlib::Record> &record_array, ByteVec &output) {
    size_t byte_size = record_array.size * csortlib::RECORD_SIZE;
    output.resize(byte_size);
    memcpy(output.data(), (const unsigned char *)record_array.ptr, byte_size);
}

void ConvertRecordArrayToBinary(
    const csortlib::Array<csortlib::Record> &record_array, ByteVec &output) {
    size_t byte_size = record_array.size * csortlib::RECORD_SIZE;
    output.resize(byte_size);
    memcpy(output.data(), (unsigned char *)record_array.ptr, byte_size);
}

void test_s3io_bin(Aws::S3::S3Client const& client,
    Aws::String const& bucket,
    Aws::String const& key) {
    
    // std::ifstream fin;
    // fin.open("part1", std::ios::in | std::ios::binary);
    // ByteVec p1;
    // fin.seekg(0, fin.end);
    // auto size = fin.tellg();
    // fin.seekg(0, fin.beg);
    // p1.resize(size);
    // fin.read((char *)p1.data(), size);
    // fin.close();
    // auto err0 = upload_file_binary(client, bucket, key, p1); 

    ByteVec part1;
    auto err = download_file_binary(client, bucket, key, part1);
    std::cout << "download_size: " << part1.size() << std::endl;
        
    std::ofstream fout;
    fout.open("part1-download", std::ios::out | std::ios::binary);
    for (auto &b : part1) {
        fout << b;
    }
    fout.close();
}
