#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#include "csortlib.h"

using namespace csortlib;

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

int main() {

  const size_t num_reducers = 10;
  const auto &boundaries = GetBoundaries(num_reducers);

  size_t num_records = 0;
  std::vector<std::string> files;

  files.push_back("./part");

  // Read number from files
  {
    for (size_t i = 0; i < files.size(); i++){
      FILE *fp = fopen(files[i].c_str(), "r");  
      if(!fp)
        perror("Failed to open file");
      fseek(fp, 0L, SEEK_END);  
      size_t fsize = ftell(fp);  
      fclose(fp); 

      if (fsize % RECORD_SIZE != 0)
        perror("File size error");

      printf("File size %lu .\n", fsize);
      num_records += fsize / RECORD_SIZE;
    }
  }

  Record *records = new Record[num_records];

  // Read file content
  {
    size_t offset = 0;
    for (size_t i = 0; i < files.size(); i++){
      FILE *fin;
      size_t file_size = 0;
      fin = fopen(files[i].c_str(), "r");
      if (fin == NULL) {
        perror("Failed to open file");
      } else {
        file_size = fread(records + offset, RECORD_SIZE, num_records, fin);
        offset += file_size;
        printf("Read %lu records.\n", file_size);
        fclose(fin);
      }
    }
  }

  const auto start1 = std::chrono::high_resolution_clock::now();
  auto ret = SortAndPartition({records, num_records}, boundaries);
  const auto stop1 = std::chrono::high_resolution_clock::now();
  printf("SortAndPartition,%ld\n\n",
          std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1)
              .count());

  // for (size_t i = 0; i < ret.size(); i++){
  //   printf("%ld %ld\n", ret[i].offset, ret[i].size);
  // }



  const auto& record_arrays = MakeConstRecordArrays(records, ret);
  const auto start2 = std::chrono::high_resolution_clock::now();
  const auto output = MergePartitions(record_arrays);
  const auto stop2 = std::chrono::high_resolution_clock::now();

  FILE* fout;
  fout = fopen("data1g-output", "w");
  if (fout == NULL) {
      perror("Failed to open file");
  } else {
      size_t writecount = fwrite(output.ptr, RECORD_SIZE, output.size, fout);
      printf("Wrote %lu bytes.\n", writecount);
      fclose(fout);
  }
  printf("Execution time (us):\n");
  printf("SortAndPartition,%ld\n",
         std::chrono::duration_cast<std::chrono::microseconds>(stop1 -
         start1)
             .count());
  printf("MergePartitions,%ld\n",
         std::chrono::duration_cast<std::chrono::microseconds>(stop2 -
         start2)
             .count());
  printf("Total,%ld\n",
         std::chrono::duration_cast<std::chrono::microseconds>(stop2 -
         start1)
             .count());

  return 0;
}
