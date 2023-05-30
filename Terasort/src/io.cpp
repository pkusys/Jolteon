#include "io.h"

std::function<std::shared_ptr<Aws::Utils::Logging::LogSystemInterface>()> GetConsoleLoggerFactory()
{
    return [] {
        return Aws::MakeShared<Aws::Utils::Logging::ConsoleLogSystem>(
            "console_logger", Aws::Utils::Logging::LogLevel::Trace);
    };
}

std::string upload_file_string(
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
        //AWS_LOGSTREAM_INFO(TAG, "Upload completed!");
        return {};
    }
    else {
        //AWS_LOGSTREAM_ERROR(TAG, "Failed with error: " << outcome.GetError());
        return outcome.GetError().GetMessage();
    }
}

std::string read_to_string(Aws::IOStream& stream, Aws::String& output)
{
    Aws::Vector<unsigned char> bytes;
    bytes.reserve(stream.tellp());
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
    Aws::String& output)
{
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
    ByteVec const& body)
{
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
        //AWS_LOGSTREAM_INFO(TAG, "Upload completed!");
        return {};
    }
    else {
        //AWS_LOGSTREAM_ERROR(TAG, "Failed with error: " << outcome.GetError());
        return outcome.GetError().GetMessage();
    }
}

std::string read_to_binary(Aws::IOStream& stream, ByteVec& output)
{
    // append to output
    char streamBuffer[1024 * 4];
    while (stream.good()) {
        stream.read(streamBuffer, sizeof(streamBuffer));
        auto bytesRead = stream.gcount();

        if (bytesRead > 0) {
            output.insert(output.end(), (unsigned char*)streamBuffer, (unsigned char*)streamBuffer + bytesRead);
        }
    }
    return {};
}

std::string download_file_binary(
    Aws::S3::S3Client const& client,
    Aws::String const& bucket,
    Aws::String const& key,
    ByteVec& output)
{
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
    for (size_t i = 0; i < record_array.size; ++i) {
        auto record = record_array.ptr + i;
        output.insert(output.end(), (const unsigned char *)record, (const unsigned char *)record + csortlib::RECORD_SIZE);
    }
}

void test_s3_io(Aws::S3::S3Client const& client,
    Aws::String const& bucket,
    Aws::String const& key) {
    
    // std::ifstream fin;
	// fin.open("../part1", std::ios::in | std::ios::binary);
	// Aws::String p1;
	// fin.seekg(0, std::ios::end);
	// p1.reserve(fin.tellg());
	// fin.seekg(0, std::ios::beg);
	// p1.assign((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
	// fin.close();
	// auto err0 = upload_file_string(client, bucket, key, p1);

	Aws::String part1;
	auto err = download_file_binary(client, bucket, key, part1);
	// write to local file
	std::ofstream fout;
	fout.open("part1-download", std::ios::out | std::ios::binary);
	fout << part1;
	fout.close();
}
