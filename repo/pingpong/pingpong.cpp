#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>
#include <assert.h>
#include <limits.h>
#include <random>

#if defined(USE_CALIPER)
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#endif

#ifndef USE_CALIPER
#define CALI_CXX_MARK_FUNCTION
#define CALI_MARK_BEGIN(...)
#define CALI_MARK_END(...)
#endif

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#endif

#if defined(USE_CUDA)
#include <cuda_runtime.h>
inline void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#endif

// ---- Operation selector (default: PingPong) ----
enum class OpKind { All, PingPong, Alltoall, Reduce, Allreduce, Bandwidth };

const char *get_hostname_for_rank(int rank, char all_hostnames[][1024], int size)
{
    if (rank >= 0 && rank < size)
        return all_hostnames[rank];
    else
        return "INVALID_RANK";
}

int extract_node_number(const char *hostname)       //helper function to build region labels
{
    int len = (int)strlen(hostname);
    int num = 0;
    int factor = 1;
    for (int i = len - 1; i >= 0; --i)
    {
        if (hostname[i] >= '0' && hostname[i] <= '9')
        {
            num += (hostname[i] - '0') * factor;
            factor *= 10;
        }
        else
        {
            break;
        }
    }
    return num;
}

// Fill buf[0..len-1] with a repeating random pattern of length 16.
void fill_with_random_pattern(char* buf, size_t len)
{
    if (!buf || len == 0)
        return;

    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 25); // 'a'..'z'

    char pattern[16];
    for (int i = 0; i < 16; ++i) {
        pattern[i] = static_cast<char>('a' + dist(gen));
    }

    for (size_t i = 0; i < len; ++i) {
        buf[i] = pattern[i % 16];
    }
}

struct RankPair {
    int src;
    int dst;
};

//building region labels specifically for 1 node (same socket and different socket)
static std::vector<RankPair>
build_pingpong_pairs(const std::string& region_label,
                     int size,
                     int sys_cores_per_socket,
                     int sys_cores_per_node,
                     int max_pairs)
{
    std::vector<RankPair> pairs;

    auto add_pair = [&](int s, int d) {
        if (s < 0 || d < 0 || s >= size || d >= size)
            return;
        pairs.push_back({s, d});
    };

    if (region_label == "Same Node Same Socket") {
        int rps = sys_cores_per_socket;
        if (rps < 2) return pairs;

        int s0 = 0;
        int s1 = rps / 8;
        int s2 = rps / 4;
        int s3 = (rps / 2) - 2;

        add_pair(s0, s0 + 1);
        add_pair(s1, s1 + 1);
        add_pair(s2, s2 + 1);
        add_pair(s3, s3 + 1);

        return pairs;
    }

    if (region_label == "Same Node Different Socket") {
        int rps = sys_cores_per_socket;
        int delta = rps;

        int s0 = 0;
        int s1 = delta / 8;
        int s2 = delta / 4;
        int s3 = (delta / 2) - 1;

        add_pair(s0, rps / 2);
        add_pair(s1, (rps / 2) + 1);
        add_pair(s2, (rps / 2) + 2);
        add_pair(s3, rps - 1);

        return pairs;
    }

    int nodes_in_comm = 0;
    {
        std::stringstream ss(region_label);
        ss >> nodes_in_comm;
    }

    if (nodes_in_comm >= 2) {
        int rpn   = sys_cores_per_node;
        int last_node_base = (nodes_in_comm - 1) * rpn;

        int offsets[4] = {
            0,
            (rpn - 1) / 4,
            (rpn - 1) / 2,
            rpn - 1
        };

        for (int off : offsets)
            add_pair(off, last_node_base + off);

        return pairs;
    }
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char my_hostname[1024];
    gethostname(my_hostname, 1023);
    my_hostname[1023] = '\0';
    char all_hostnames[size][1024];
    MPI_Gather(my_hostname, 1024, MPI_CHAR, all_hostnames, 1024, MPI_CHAR, 0,
               MPI_COMM_WORLD);

#if defined(USE_CALIPER)
    std::vector<std::string> all_comm_pairs;
    static std::map<int, cali::ConfigManager> mgr;
    MPI_Comm adiak_comm = MPI_COMM_WORLD;
    adiak::init(&adiak_comm);
    adiak::collect_all();
    CALI_CXX_MARK_FUNCTION;
#endif

    int num_iterations = 10;
    const int WINDOW_SIZE = 64; // OSU-style window exchange; no SINGLE/MULTIPLE modes
    int msg_size = 1;
    int n_nodes = 1;
    int sys_cores_per_socket = 1;
    int sys_cores_per_node = 1;
    std::string metadata;
    const char *warmup_region = "warmup";
    const char *warmup_region_aa = "warmup_aa";
    const char *warmup_region_red = "warmup_red";
    const char *warmup_region_ar = "warmup_ar";
    int pingpong_num_pairs = 1;

    // ---- default to PingPong ----
    OpKind op = OpKind::PingPong;

    int opt;
    const char *usage =
        "Usage: %s [-h] [-i n-iterations] [-p rank1,rank2] [-m msg_sz] "
        "[-n n_nodes] [-s sys_cores_per_socket] [-c sys_cores_per_node] [-b metadata] [-k pingpong_num_pairs]"
        "[-O pingpong|alltoall|reduce|allreduce|bandwidth|all]\n"
        "Default: -O pingpong\n";

    while ((opt = getopt(argc, argv, "hi:p:m:n:s:c:b:k:O:")) != -1)
    {
        switch (opt)
        {
            case 'h':
                printf(usage, argv[0]);
                MPI_Finalize();
                return 0;
            case 'i':
                num_iterations = atoi(optarg);
                break;
            case 'p':
                // kept for compatibility; parse partners here if desired
                break;
            case 'm':
                msg_size = atoi(optarg);
                break;
            case 'n':
                n_nodes = atoi(optarg);
                break;
            case 's':
                sys_cores_per_socket = atoi(optarg);
                break;
            case 'c':
                sys_cores_per_node = atoi(optarg);
                break;
            case 'b':
                metadata = optarg;
                break;
            case 'k':
                pingpong_num_pairs = atoi(optarg);
                break;
            case 'O':
            {
                std::string s = optarg ? std::string(optarg) : std::string();
                if (s == "pingpong")      op = OpKind::PingPong;
                else if (s == "alltoall") op = OpKind::Alltoall;
                else if (s == "reduce")   op = OpKind::Reduce;
                else if (s == "allreduce")op = OpKind::Allreduce;
                else if (s == "bandwidth")op = OpKind::Bandwidth;
                else if (s == "all")      op = OpKind::All;
                else {
                    if (rank == 0)
                        fprintf(stderr, "Unknown -O value '%s'. Expected pingpong|alltoall|reduce|allreduce|bandwidth|all\n", s.c_str());
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                break;
            }
            default:
                if (rank == 0) printf(usage, argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (rank == 0)
    {
        printf("Configuration:\n");
        printf("Number iterations: %d\n", num_iterations);            //renamed for easier understanding
        printf("Message size: %d bytes\n", msg_size);
        printf("Cores per socket: %d\n", sys_cores_per_socket);
        printf("Cores per node: %d\n", sys_cores_per_node);
        printf("Nodes: %d\n", n_nodes);
        printf("World size: %d\n", size);
        printf("Pingpong num pairs: %d\n", pingpong_num_pairs);       //only for Pingpong
        printf("Mode (-O): %s\n",
               op == OpKind::PingPong ? "pingpong" :
               op == OpKind::Alltoall ? "alltoall" :
               op == OpKind::Reduce   ? "reduce" :
               op == OpKind::Allreduce? "allreduce" : 
               op == OpKind::Bandwidth? "bandwidth" : "all");

#if defined(USE_CALIPER)
        std::stringstream rankmap;
        rankmap << "{";
        for (int i = 0; i < size; ++i)
        {
            rankmap << "\"" << i << "\": \"" << all_hostnames[i] << "\"";
            if (i < size - 1)
                rankmap << ", ";
        }
        rankmap << "}";
        adiak::value("rank_node_map", rankmap.str());
        adiak::value("iterations", num_iterations);
        adiak::value("pingpong_num_pairs", pingpong_num_pairs);
#endif
    }

#if defined(USE_CALIPER)
    cali_id_t src_rank_attr = cali_create_attribute("src_rank", CALI_TYPE_INT,
                              CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t dest_rank_attr = cali_create_attribute("dest_rank", CALI_TYPE_INT,
                               CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t src_node_attr = cali_create_attribute("src_node", CALI_TYPE_INT,
                              CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t dest_node_attr = cali_create_attribute("dest_node", CALI_TYPE_INT,
                               CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t message_size_attr = cali_create_attribute("message_size_bytes",
                                  CALI_TYPE_INT, CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t comm_phase_attr = cali_create_attribute("comm_phase", CALI_TYPE_STRING,
                                CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t aa_avg_time_sec_attr = cali_create_attribute("aa_avg_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t aa_max_time_sec_attr = cali_create_attribute("aa_max_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t aa_min_time_sec_attr = cali_create_attribute("aa_min_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t pp_avg_time_sec_attr = cali_create_attribute("pp_avg_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t pp_max_time_sec_attr = cali_create_attribute("pp_max_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t pp_min_time_sec_attr = cali_create_attribute("pp_min_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t red_avg_time_sec_attr = cali_create_attribute("red_avg_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t red_max_time_sec_attr = cali_create_attribute("red_max_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t red_min_time_sec_attr = cali_create_attribute("red_min_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t ar_avg_time_sec_attr  = cali_create_attribute("ar_avg_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t ar_max_time_sec_attr  = cali_create_attribute("ar_max_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t ar_min_time_sec_attr  = cali_create_attribute("ar_min_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t bw_avg_time_sec_attr  = cali_create_attribute("bw_avg_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t bw_max_time_sec_attr  = cali_create_attribute("bw_max_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);
    cali_id_t bw_min_time_sec_attr  = cali_create_attribute("bw_min_time_sec", CALI_TYPE_DOUBLE,
                                 CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE);

    const char *src_dest_attributes = R"json(
        {
            "name": "pingpong_attributes",
            "type": "boolean",
            "category": "metric",
            "description": "Collect pingpong attributes",
            "query":
            [
            {
                "level": "local",
                "select":
                [
                {"expr": "any(max#src_rank)", "as": "src_rank"},
                {"expr": "any(max#dest_rank)", "as": "dest_rank"},
                {"expr": "any(max#src_node)", "as": "src_node"},
                {"expr": "any(max#dest_node)", "as": "dest_node"},
                {"expr": "any(max#message_size_bytes)", "as": "message_size_bytes"},
                {"expr": "any(max#aa_avg_time_sec)", "as" : "aa_avg_s"},
                {"expr": "any(max#aa_max_time_sec)", "as" : "aa_max_s"},
                {"expr": "any(max#aa_min_time_sec)", "as" : "aa_min_s"},
                {"expr": "any(max#pp_avg_time_sec)", "as" : "pp_avg_s"},
                {"expr": "any(max#pp_max_time_sec)", "as" : "pp_max_s"},
                {"expr": "any(max#pp_min_time_sec)", "as" : "pp_min_s"},
                {"expr": "any(max#red_avg_time_sec)", "as" : "red_avg_s"},
                {"expr": "any(max#red_max_time_sec)", "as" : "red_max_s"},
                {"expr": "any(max#red_min_time_sec)", "as" : "red_min_s"},
                {"expr": "any(max#ar_avg_time_sec)", "as" : "ar_avg_s"},
                {"expr": "any(max#ar_max_time_sec)", "as" : "ar_max_s"},
                {"expr": "any(max#ar_min_time_sec)", "as" : "ar_min_s"},
                {"expr": "any(max#bw_avg_time_sec)", "as" : "bw_avg_s"},
                {"expr": "any(max#bw_max_time_sec)", "as" : "bw_max_s"},
                {"expr": "any(max#bw_min_time_sec)", "as" : "bw_min_s"}
                ],
                "group by": ["comm_phase"],
            },
            {
                "level": "cross",
                "select":
                [
                {"expr": "any(any#max#src_rank)", "as": "src_rank"},
                {"expr": "any(any#max#dest_rank)", "as": "dest_rank"},
                {"expr": "any(any#max#src_node)", "as": "src_node"},
                {"expr": "any(any#max#dest_node)", "as": "dest_node"},
                {"expr": "any(any#max#message_size_bytes)", "as": "message_size_bytes"},
                {"expr": "any(any#max#aa_avg_time_sec)", "as" : "aa_avg_s"},
                {"expr": "any(any#max#aa_max_time_sec)", "as" : "aa_max_s"},
                {"expr": "any(any#max#aa_min_time_sec)", "as" : "aa_min_s"},
                {"expr": "any(any#max#pp_avg_time_sec)", "as" : "pp_avg_s"},
                {"expr": "any(any#max#pp_max_time_sec)", "as" : "pp_max_s"},
                {"expr": "any(any#max#pp_min_time_sec)", "as" : "pp_min_s"},
                {"expr": "any(any#max#red_avg_time_sec)", "as" : "red_avg_s"},
                {"expr": "any(any#max#red_max_time_sec)", "as" : "red_max_s"},
                {"expr": "any(any#max#red_min_time_sec)", "as" : "red_min_s"},
                {"expr": "any(any#max#ar_avg_time_sec)", "as" : "ar_avg_s"},
                {"expr": "any(any#max#ar_max_time_sec)", "as" : "ar_max_s"},
                {"expr": "any(any#max#ar_min_time_sec)", "as" : "ar_min_s"},
                {"expr": "any(any#max#bw_avg_time_sec)", "as" : "bw_avg_s"},
                {"expr": "any(any#max#bw_max_time_sec)", "as" : "bw_max_s"},
                {"expr": "any(any#max#bw_min_time_sec)", "as" : "bw_min_s"}
                ],
                "group by": ["comm_phase"],
            }
            ]
        }
        )json";
#endif

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&now_time), "%Y%m%d_%H%M%S");

    int P = sys_cores_per_node * n_nodes;
    std::vector<int> partners;
    std::map<int, std::string> region_names;

    int current_nodes = n_nodes;
    int current_p = P;

    while (current_p > 2)                               //splitting for region labels for pingpong
    {
        int partner_rank = current_p - 1;
        std::string label;
        if (current_nodes >= 2)
            label = std::to_string(current_nodes) + " nodes";
        else
            label = "Same Node Different Socket";

        if (partner_rank > 0 && partner_rank < size)
        {
            partners.push_back(partner_rank);
            region_names[partner_rank] = label;
        }
        current_nodes = current_nodes / 2;
        current_p = current_nodes * sys_cores_per_node;
    }

    // Always add same node same socket as 0 <-> 1
    if (1 < size)
    {
        partners.push_back(1);
        region_names[1] = "Same Node Same Socket";
    }

    for (int message = msg_size; message <= pow(msg_size, 1); message *= 8)
    {
#if defined(USE_CALIPER)
        std::string profile = "spot(output=" + std::to_string(message) + "_" +
                              timestamp.str() +
                              ".cali, profile.mpi),metadata(file=" + metadata +
                              "),metadata(file=/etc/node_info.json,keys=\"host.os\")";

        cali_set_int(message_size_attr, message);

        mgr[message].add_option_spec(src_dest_attributes);
        mgr[message].set_default_parameter("pingpong_attributes", "true");
        adiak::value("message_size", message);
        mgr[message].add(profile.c_str());
        mgr[message].start();
#endif

        // ===================== PINGPONG =====================
        if (op == OpKind::PingPong || op == OpKind::All)
        {
            for (int partner_rank : partners)
            {
                std::string region_label = region_names[partner_rank];

                // Build up to N pairs for this region
                std::vector<RankPair> pairs =
                    build_pingpong_pairs(region_label,
                                         size,
                                         sys_cores_per_socket,
                                         sys_cores_per_node,
                                         pingpong_num_pairs);

                if (pairs.empty()) {
                    if (rank == 0)
                        printf("Skipping region %s: no valid pingpong pairs\n",
                               region_label.c_str());
                    continue;
                }

                // Map each rank to its partner (or MPI_PROC_NULL if not in any pair)
                std::vector<int> my_partner(size, MPI_PROC_NULL);
                for (auto &p : pairs) {
                    my_partner[p.src] = p.dst;
                    my_partner[p.dst] = p.src;
                }

                int partner = my_partner[rank];

                //split communicator so only paired ranks participate
                const int active = (partner != MPI_PROC_NULL);
                MPI_Comm pp_comm = MPI_COMM_NULL;
                MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, rank, &pp_comm);

                if (!active) {
                    //Not in any pair for this region: do NOT touch barriers inside this region
                    continue;
                }

                if (rank == 0)
                {
                    printf("\n--- Testing %s (PINGPONG) with %zu pairs --- (partner rank: %d) \n",
                           region_label.c_str(), pairs.size(), partner_rank);
                    for (auto &p : pairs) {
                        printf("  pair %d (%s) <-> %d (%s)\n",
                               p.src, all_hostnames[p.src],
                               p.dst, all_hostnames[p.dst]);
                    }
                    fflush(stdout);

#if defined(USE_CALIPER)
                    // Use the first pair as representative for Caliper attributes
                    cali_set_int(src_rank_attr, pairs[0].src);
                    cali_set_int(dest_rank_attr, pairs[0].dst);
                    cali_set_int(src_node_attr,
                                 extract_node_number(all_hostnames[pairs[0].src]));
                    cali_set_int(dest_node_attr,
                                 extract_node_number(all_hostnames[pairs[0].dst]));
#endif
                }

                double total_time = 0.0;
                int warmup = 1;

                const int base_tag = 1000 + ((partner_rank % 2000) * 10);
                const int TAG_A = base_tag + 0;
                const int TAG_B = base_tag + 1;

                // OSU-style directional tags for CPU path
                const int my_recv_tag = (rank < partner) ? TAG_A : TAG_B;
                const int my_send_tag = (rank < partner) ? TAG_B : TAG_A;

                // ---------- buffer allocation ----------
#if defined(USE_HIP)
                char *send_flat = nullptr;
                char *recv_flat = nullptr;

                size_t pp_total_bytes = (size_t)WINDOW_SIZE * (size_t)message;

                hipError_t err1 = hipMalloc((void**)&send_flat, pp_total_bytes);
                hipError_t err2 = hipMalloc((void**)&recv_flat, pp_total_bytes);

                if (err1 != hipSuccess || err2 != hipSuccess) {
                    fprintf(stderr, "HIP malloc failed: %s %s\n", hipGetErrorString(err1), hipGetErrorString(err2));
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                char *h_tmp = (char*)malloc(pp_total_bytes);
                if (!h_tmp) {
                    fprintf(stderr, "Rank %d host malloc failed for %zu bytes\n", rank, pp_total_bytes);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                fill_with_random_pattern(h_tmp, pp_total_bytes);

                hipError_t cuerr1 = hipMemcpy(send_flat, h_tmp, pp_total_bytes, hipMemcpyHostToDevice);
                assert(cuerr1 == hipSuccess);

                hipError_t cuerr2 = hipMemset(recv_flat, 0, pp_total_bytes);
                assert(cuerr2 == hipSuccess);

                free(h_tmp);

                std::vector<char*> s_buf(WINDOW_SIZE);
                std::vector<char*> r_buf(WINDOW_SIZE);

                for (int j = 0; j < WINDOW_SIZE; ++j) {
                    s_buf[j] = send_flat + (size_t)j * (size_t)message;
                    r_buf[j] = recv_flat + (size_t)j * (size_t)message;
                }

                std::vector<MPI_Request> send_request(WINDOW_SIZE);
                std::vector<MPI_Request> recv_request(WINDOW_SIZE);

#elif defined(USE_CUDA)
                int dev_count = 0;
                cuda_check(cudaGetDeviceCount(&dev_count));
                cuda_check(cudaSetDevice(rank % (dev_count > 0 ? dev_count : 1)));

                char *d_send = nullptr;
                char *d_recv = nullptr;
                cuda_check(cudaMalloc((void**)&d_send, message));
                cuda_check(cudaMalloc((void**)&d_recv, message));

                char *h_send = nullptr, *h_recv = nullptr;
                cuda_check(cudaMallocHost((void**)&h_send, message));
                cuda_check(cudaMallocHost((void**)&h_recv, message));

                fill_with_random_pattern(h_send, (size_t)message);
                memset(h_recv, 0, message);

                cuda_check(cudaMemcpy(d_send, h_send, message, cudaMemcpyHostToDevice));
                cuda_check(cudaMemset(d_recv, 0, message));

#else
                char *send_flat = (char*)malloc((size_t)WINDOW_SIZE * (size_t)message);
                char *recv_flat = (char*)malloc((size_t)WINDOW_SIZE * (size_t)message);

                std::vector<char*> s_buf(WINDOW_SIZE);
                std::vector<char*> r_buf(WINDOW_SIZE);

                for (int j = 0; j < WINDOW_SIZE; ++j) {
                    s_buf[j] = send_flat + (size_t)j * (size_t)message;
                    r_buf[j] = recv_flat + (size_t)j * (size_t)message;
                    fill_with_random_pattern(s_buf[j], (size_t)message);
                    memset(r_buf[j], 0, (size_t)message);
                }

                std::vector<MPI_Request> send_request(WINDOW_SIZE);
                std::vector<MPI_Request> recv_request(WINDOW_SIZE);
                std::vector<MPI_Request> reqs(2 * WINDOW_SIZE);
#endif

                // Sync ONLY participating ranks
                MPI_Barrier(pp_comm);

                // ---------- warmup ----------
#if defined(USE_CALIPER)
                CALI_MARK_BEGIN(warmup_region);
#endif
                for (int i = 0; i < warmup; i++)
                {
#if defined(USE_HIP)
                    for (int j = 0; j < WINDOW_SIZE; ++j) {
                        MPI_Irecv(r_buf[j], message, MPI_CHAR, partner, my_recv_tag,
                                MPI_COMM_WORLD, &recv_request[j]);
                    }
                    for (int j = 0; j < WINDOW_SIZE; ++j) {
                        MPI_Isend(s_buf[j], message, MPI_CHAR, partner, my_send_tag,
                                MPI_COMM_WORLD, &send_request[j]);
                    }
                    MPI_Waitall(WINDOW_SIZE, send_request.data(), MPI_STATUSES_IGNORE);
                    MPI_Waitall(WINDOW_SIZE, recv_request.data(), MPI_STATUSES_IGNORE);
#elif defined(USE_CUDA)
                    if (rank < partner) {
                        cuda_check(cudaMemcpy(h_send, d_send, message, cudaMemcpyDeviceToHost));
                        MPI_Send(h_send, message, MPI_CHAR, partner, TAG_A, MPI_COMM_WORLD);
                        MPI_Recv(h_recv, message, MPI_CHAR, partner, TAG_A, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                        cuda_check(cudaMemcpy(d_recv, h_recv, message, cudaMemcpyHostToDevice));
                    } else {
                        MPI_Recv(h_recv, message, MPI_CHAR, partner, TAG_A, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                        cuda_check(cudaMemcpy(d_recv, h_recv, message, cudaMemcpyHostToDevice));
                        cuda_check(cudaMemcpy(h_send, d_send, message, cudaMemcpyDeviceToHost));
                        MPI_Send(h_send, message, MPI_CHAR, partner, TAG_A, MPI_COMM_WORLD);
                    }
#else
                    for (int j = 0; j < WINDOW_SIZE; ++j) {
                        MPI_Irecv(r_buf[j], message, MPI_CHAR, partner, my_recv_tag,
                                  MPI_COMM_WORLD, &recv_request[j]);
                    }
                    for (int j = 0; j < WINDOW_SIZE; ++j) {
                        MPI_Isend(s_buf[j], message, MPI_CHAR, partner, my_send_tag,
                                  MPI_COMM_WORLD, &send_request[j]);
                    }
                    MPI_Waitall(WINDOW_SIZE, send_request.data(), MPI_STATUSES_IGNORE);
                    MPI_Waitall(WINDOW_SIZE, recv_request.data(), MPI_STATUSES_IGNORE);
#endif
                }

#if defined(USE_CALIPER)
                CALI_MARK_END(warmup_region);
                CALI_MARK_BEGIN(region_label.c_str());
#endif

                // ---------- timed ping-pong ----------
                double min_rtt = std::numeric_limits<double>::infinity();
                double max_rtt = 0.0;
                int iters = 0;

                bool i_am_timing_rank = (rank < partner);

                MPI_Barrier(pp_comm);

                for (int i = 0; i < num_iterations; i++)
                {
                    double start = 0.0, end = 0.0;

                    if (i_am_timing_rank)
                        start = MPI_Wtime();

#if defined(USE_HIP)
                    for (int j = 0; j < WINDOW_SIZE; ++j) {
                        MPI_Irecv(r_buf[j], message, MPI_CHAR, partner, my_recv_tag,
                                MPI_COMM_WORLD, &recv_request[j]);
                    }
                    for (int j = 0; j < WINDOW_SIZE; ++j) {
                        MPI_Isend(s_buf[j], message, MPI_CHAR, partner, my_send_tag,
                                MPI_COMM_WORLD, &send_request[j]);
                    }
                    MPI_Waitall(WINDOW_SIZE, send_request.data(), MPI_STATUSES_IGNORE);
                    MPI_Waitall(WINDOW_SIZE, recv_request.data(), MPI_STATUSES_IGNORE);
#elif defined(USE_CUDA)
                    if (rank < partner) {
                        cuda_check(cudaMemcpy(h_send, d_send, message, cudaMemcpyDeviceToHost));
                        MPI_Send(h_send, message, MPI_CHAR, partner, TAG_B, MPI_COMM_WORLD);
                        MPI_Recv(h_recv, message, MPI_CHAR, partner, TAG_B, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                        cuda_check(cudaMemcpy(d_recv, h_recv, message, cudaMemcpyHostToDevice));
                    } else {
                        MPI_Recv(h_recv, message, MPI_CHAR, partner, TAG_B, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                        cuda_check(cudaMemcpy(d_recv, h_recv, message, cudaMemcpyHostToDevice));
                        cuda_check(cudaMemcpy(h_send, d_send, message, cudaMemcpyDeviceToHost));
                        MPI_Send(h_send, message, MPI_CHAR, partner, TAG_B, MPI_COMM_WORLD);
                    }
#else
                    for (int j = 0; j < WINDOW_SIZE; ++j) {
                        MPI_Irecv(r_buf[j], message, MPI_CHAR, partner, my_recv_tag,
                                  MPI_COMM_WORLD, &recv_request[j]);
                    }
                    for (int j = 0; j < WINDOW_SIZE; ++j) {
                        MPI_Isend(s_buf[j], message, MPI_CHAR, partner, my_send_tag,
                                  MPI_COMM_WORLD, &send_request[j]);
                    }
                    MPI_Waitall(WINDOW_SIZE, send_request.data(), MPI_STATUSES_IGNORE);
                    MPI_Waitall(WINDOW_SIZE, recv_request.data(), MPI_STATUSES_IGNORE);
#endif

                    // calculation for round-trip time
                    if (i_am_timing_rank) {
                        end = MPI_Wtime();
                        double rtt = end - start;
                        total_time += rtt;
                        if (rtt < min_rtt) min_rtt = rtt;
                        if (rtt > max_rtt) max_rtt = rtt;
                        ++iters;
                    }
                }

                MPI_Barrier(pp_comm);

                if (i_am_timing_rank)
                {
                    double avg_rtt = (iters > 0) ? (total_time / iters) : 0.0;

#if defined(USE_CALIPER)
                    cali_set_string(comm_phase_attr, "pingpong");
                    cali_set_double(pp_avg_time_sec_attr, avg_rtt);
                    cali_set_double(pp_max_time_sec_attr, max_rtt);
                    cali_set_double(pp_min_time_sec_attr, min_rtt);
#endif

                    printf("PINGPONG %s pair (%d,%d): avg=%g s, min=%g s, max=%g s\n",
                           region_label.c_str(), rank, partner,
                           avg_rtt, min_rtt, max_rtt);
                    fflush(stdout);
                }

#if defined(USE_CALIPER)
                CALI_MARK_END(region_label.c_str());
#endif

                // ---------- cleanup ----------
#if defined(USE_HIP)
                hipFree(send_flat);
                hipFree(recv_flat);
#elif defined(USE_CUDA)
                cuda_check(cudaFreeHost(h_send));
                cuda_check(cudaFreeHost(h_recv));
                cuda_check(cudaFree(d_send));
                cuda_check(cudaFree(d_recv));
#else
                free(send_flat);
                free(recv_flat);
#endif

                MPI_Comm_free(&pp_comm);

            } // end for (partner_rank : partners)

            if (rank == 0) printf("Done with Pingpong\n");
        }     // end if (PingPong || All)

        MPI_Barrier(MPI_COMM_WORLD);

        // ===================== ALLTOALL =====================
        if (op == OpKind::Alltoall || op == OpKind::All)
        {
            for (int partner_rank : partners)
            {
                std::string region_label = region_names[partner_rank];

                int ranks_in_region = partner_rank + 1;
                int active = (rank < ranks_in_region);

                MPI_Comm region_comm = MPI_COMM_NULL;
                MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, rank, &region_comm);

                if (!active)
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                    continue;
                }

                int region_rank = 0;
                int region_size = 0;
                MPI_Comm_rank(region_comm, &region_rank);
                MPI_Comm_size(region_comm, &region_size);

                size_t aa_bytes_per_rank = static_cast<size_t>(message);
                size_t aa_total_bytes = aa_bytes_per_rank * static_cast<size_t>(region_size);

                int warmup = 1;
                double alltoall_total_time = 0.0;

                if (region_rank == 0) {
                    printf("\n--- Testing %s (ALLTOALL) with %d ranks ---\n", region_label.c_str(), region_size);
                    printf("message=%d bytes, total buffer per rank=%zu bytes\n", message, aa_total_bytes);
                    fflush(stdout);
                }

                // --- buffer allocation ---
#if defined(USE_HIP)
                char *aa_send_dev;
                char *aa_recv_dev;

                hipError_t a_err1 = hipMalloc((void**)&aa_send_dev, aa_total_bytes);
                hipError_t a_err2 = hipMalloc((void**)&aa_recv_dev, aa_total_bytes);

                if (a_err1 != hipSuccess || a_err2 != hipSuccess) {
                    fprintf(stderr, "HIP malloc failed: %s %s\n", hipGetErrorString(a_err1), hipGetErrorString(a_err2));
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                char *aa_send_host = (char*) malloc(aa_total_bytes);
                char *aa_recv_host = (char*) malloc(aa_total_bytes);
                fill_with_random_pattern(aa_send_host, aa_total_bytes);
                memset(aa_recv_host, 0, aa_total_bytes);

                hipError_t a_cuerr1 = hipMemcpy(aa_send_dev, aa_send_host, aa_total_bytes, hipMemcpyHostToDevice);
                assert(a_cuerr1 == hipSuccess);
                hipError_t a_cuerr2 = hipMemset(aa_recv_dev, 0, aa_total_bytes);
                assert(a_cuerr2 == hipSuccess);

#elif defined(USE_CUDA)
                int a_dev_count = 0;
                cuda_check(cudaGetDeviceCount(&a_dev_count));
                cuda_check(cudaSetDevice(rank % (a_dev_count > 0 ? a_dev_count : 1)));

                char *ad_send = nullptr;
                char *ad_recv = nullptr;
                cuda_check(cudaMalloc((void**)&ad_send, aa_total_bytes));
                cuda_check(cudaMalloc((void**)&ad_recv, aa_total_bytes));

                char *ah_send = nullptr;
                char *ah_recv = nullptr;
                cuda_check(cudaMallocHost((void**)&ah_send, aa_total_bytes));
                cuda_check(cudaMallocHost((void**)&ah_recv, aa_total_bytes));

                fill_with_random_pattern(ah_send, aa_total_bytes);
                memset(ah_recv, 0, aa_total_bytes);

                cuda_check(cudaMemcpy(ad_send, ah_send, aa_total_bytes, cudaMemcpyHostToDevice));
                cuda_check(cudaMemset(ad_recv, 0, aa_total_bytes));

#else
                char *aa_send = (char*) malloc(aa_total_bytes);
                char *aa_recv = (char*) malloc(aa_total_bytes);
                fill_with_random_pattern(aa_send, aa_total_bytes);
                memset(aa_recv,  0, aa_total_bytes);
#endif

                // --- begin warmup ---
#if defined(USE_CALIPER)
                CALI_MARK_BEGIN(warmup_region_aa);
#endif
                printf("rank %d: begin warmup\n", rank);
                fflush(stdout);

                for (int i = 0; i < warmup; i++)
                {
#if defined(USE_HIP)
                    // hipMemcpy(aa_send_host, aa_send_dev, aa_total_bytes, hipMemcpyDeviceToHost);
                    // hipDeviceSynchronize();
                    // MPI_Alltoall(aa_send_host, (int)aa_bytes_per_rank, MPI_CHAR, aa_recv_host, (int)aa_bytes_per_rank, MPI_CHAR, region_comm);
                    // hipMemcpy(aa_recv_dev, aa_recv_host, aa_total_bytes, hipMemcpyHostToDevice);
                    // hipDeviceSynchronize();
                    MPI_Alltoall(aa_send_dev, (int)aa_bytes_per_rank, MPI_CHAR, aa_recv_dev, (int)aa_bytes_per_rank, MPI_CHAR, region_comm);
#elif defined(USE_CUDA)
                    cuda_check(cudaMemcpy(ah_send, ad_send, aa_total_bytes, cudaMemcpyDeviceToHost));
                    MPI_Alltoall(ah_send, (int)aa_bytes_per_rank, MPI_CHAR, ah_recv, (int)aa_bytes_per_rank, MPI_CHAR, region_comm);
                    cuda_check(cudaMemcpy(ad_recv, ah_recv, aa_total_bytes, cudaMemcpyHostToDevice));
#else
                    MPI_Alltoall(aa_send, (int)aa_bytes_per_rank, MPI_CHAR, aa_recv, (int)aa_bytes_per_rank, MPI_CHAR, region_comm);
#endif
                }

#if defined(USE_CALIPER)
                CALI_MARK_END(warmup_region_aa);
                CALI_MARK_BEGIN(region_label.c_str());
#endif

                printf("rank %d: end warmup, begin timing\n", rank);
                fflush(stdout);
                double min_rtt = std::numeric_limits<double>::infinity();
                double max_rtt = 0.0;
                int iters = 0;

                // --- timed alltoall ---
                MPI_Barrier(region_comm);

                double t0 = MPI_Wtime();

                for (int it = 0; it < num_iterations; ++it)
                {
#if defined(USE_HIP)
                    // hipMemcpy(aa_send_host, aa_send_dev, aa_total_bytes, hipMemcpyDeviceToHost);
                    // hipDeviceSynchronize();
                    // MPI_Alltoall(aa_send_host, (int)aa_bytes_per_rank, MPI_CHAR, aa_recv_host, (int)aa_bytes_per_rank, MPI_CHAR, region_comm);
                    // hipMemcpy(aa_recv_dev, aa_recv_host, aa_total_bytes, hipMemcpyHostToDevice);
                    // hipDeviceSynchronize();
                    MPI_Alltoall(aa_send_dev, (int)aa_bytes_per_rank, MPI_CHAR, aa_recv_dev, (int)aa_bytes_per_rank, MPI_CHAR, region_comm);
#elif defined(USE_CUDA)
                    cuda_check(cudaMemcpy(ah_send, ad_send, aa_total_bytes, cudaMemcpyDeviceToHost));
                    MPI_Alltoall(ah_send, (int)aa_bytes_per_rank, MPI_CHAR, ah_recv, (int)aa_bytes_per_rank, MPI_CHAR, region_comm);
                    cuda_check(cudaMemcpy(ad_recv, ah_recv, aa_total_bytes, cudaMemcpyHostToDevice));
#else
                    MPI_Alltoall(aa_send, (int)aa_bytes_per_rank, MPI_CHAR, aa_recv, (int)aa_bytes_per_rank, MPI_CHAR, region_comm);
#endif
                }

                double local_total_time = MPI_Wtime() - t0;

                double max_total_time = 0.0;
                MPI_Reduce(&local_total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, region_comm);

                if (region_rank == 0)
                {
                    double avg_time = max_total_time / num_iterations;

#if defined(USE_CALIPER)
                    cali_set_string(comm_phase_attr, "alltoall");
                    cali_set_double(aa_avg_time_sec_attr, avg_time);
                    cali_set_double(aa_max_time_sec_attr, max_total_time);      //max total time for # iterations
                    //cali_set_double(aa_min_time_sec_attr, avg_time);
#endif
                    printf("ALLTOALL %s: total=%g s, avg_per_iter=%g s over %d iterations\n",
                        region_label.c_str(), max_total_time, avg_time, num_iterations);
                    fflush(stdout);
                }
#if defined(USE_CALIPER)
                CALI_MARK_END(region_label.c_str());
#endif
                printf("rank %d: end timing\n", rank);
                fflush(stdout);

                // --- freeing up memory ---
#if defined(USE_HIP)
                free(aa_send_host);
                free(aa_recv_host);
                hipFree(aa_send_dev);
                hipFree(aa_recv_dev);
#elif defined(USE_CUDA)
                cuda_check(cudaFreeHost(ah_send));
                cuda_check(cudaFreeHost(ah_recv));
                cuda_check(cudaFree(ad_send));
                cuda_check(cudaFree(ad_recv));
#else
                free(aa_send);
                free(aa_recv);
#endif
                printf("freed memory\n");
                fflush(stdout);
                MPI_Comm_free(&region_comm);        //free subcommunicator
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // ===================== REDUCE =====================
        if (op == OpKind::Reduce || op == OpKind::All)
        {
            for (int partner_rank : partners)
            {
                std::string region_label = region_names[partner_rank];

                int ranks_in_region = partner_rank + 1;
                int active = (rank < ranks_in_region);
        
                MPI_Comm region_comm = MPI_COMM_NULL;
                MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, rank, &region_comm);

                if (!active)
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                    continue;
                }
        
                int region_rank = 0;
                int region_size = 0;
                MPI_Comm_rank(region_comm, &region_rank);
                MPI_Comm_size(region_comm, &region_size);

                size_t red_count = static_cast<size_t>(message);
                if (red_count > INT_MAX) {
                    if (rank == 0) fprintf(stderr, "Reduce count too large\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                int warmup = 1;
                double red_total_time = 0.0;

                // --- buffer allocation ---
#if defined(USE_CUDA)
                char *rd_d_send=nullptr;
                char *rd_d_recv=nullptr;

                cuda_check(cudaMalloc((void**)&rd_d_send, red_count));
                cuda_check(cudaMalloc((void**)&rd_d_recv, red_count));

                char *rd_h_send=nullptr;
                char *rd_h_recv=nullptr;
                cuda_check(cudaMallocHost((void**)&rd_h_send, red_count));
                cuda_check(cudaMallocHost((void**)&rd_h_recv, red_count));

                fill_with_random_pattern(rd_h_send, red_count);
                memset(rd_h_recv, 0,   red_count);

                cuda_check(cudaMemcpy(rd_d_send, rd_h_send, red_count, cudaMemcpyHostToDevice));
                cuda_check(cudaMemset(rd_d_recv, 0,   red_count));

#elif defined(USE_HIP)
                char *rd_d_send=nullptr;
                char *rd_d_recv=nullptr;

                hipError_t a_err1 = hipMalloc((void**)&rd_d_send, red_count);
                hipError_t a_err2 = hipMalloc((void**)&rd_d_recv, red_count);

                if (a_err1 != hipSuccess || a_err2 != hipSuccess) {
                    fprintf(stderr, "HIP malloc failed: %s %s\n", hipGetErrorString(a_err1), hipGetErrorString(a_err2));
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                char *rd_h_send = (char*)malloc(red_count);
                char *rd_h_recv = (char*)malloc(red_count);
                fill_with_random_pattern(rd_h_send, red_count);
                memset(rd_h_recv, 0,   red_count);

                hipError_t a_cuerr1 = hipMemcpy(rd_d_send, rd_h_send, red_count, hipMemcpyHostToDevice);
                assert(a_cuerr1 == hipSuccess);
                hipError_t a_cuerr2 = hipMemset(rd_d_recv, 0, red_count);
                assert(a_cuerr2 == hipSuccess);
#else
                char *rd_send = (char*)malloc(red_count);
                char *rd_recv = (char*)malloc(red_count);
                fill_with_random_pattern(rd_send, red_count);
                memset(rd_recv, 0,   red_count);
#endif

                // --- warmup section ---
#if defined(USE_CALIPER)
                CALI_MARK_BEGIN(warmup_region_red);
#endif
                printf("rank %d: begin warmup\n", rank);
                fflush(stdout);

                for (int i = 0; i < warmup; i++)
                {
#if defined(USE_CUDA)
                    cuda_check(cudaMemcpy(rd_h_send, rd_d_send, red_count, cudaMemcpyDeviceToHost));
                    MPI_Reduce(rd_h_send, rd_h_recv, (int)red_count, MPI_CHAR, MPI_SUM, 0, region_comm);
                    cuda_check(cudaMemcpy(rd_d_recv, rd_h_recv, red_count, cudaMemcpyHostToDevice));
#elif defined(USE_HIP)
                    // hipMemcpy(rd_h_send, rd_d_send, red_count, hipMemcpyDeviceToHost);
                    // MPI_Reduce(rd_h_send, rd_h_recv, (int)red_count, MPI_CHAR, MPI_SUM, 0, MPI_COMM_WORLD);
                    // hipMemcpy(rd_d_recv, rd_h_recv, red_count, hipMemcpyHostToDevice);
                    MPI_Reduce(rd_d_send, rd_d_recv, (int)red_count, MPI_CHAR, MPI_SUM, 0, region_comm);
#else
                    MPI_Reduce(rd_send, rd_recv, (int)red_count, MPI_CHAR, MPI_SUM, 0, region_comm);
#endif
                }

#if defined(USE_CALIPER)
                CALI_MARK_END(warmup_region_red);
                CALI_MARK_BEGIN(region_label.c_str());
#endif
                printf("rank %d: end warmup, begin timing\n", rank);
                fflush(stdout);

                double min_rtt = std::numeric_limits<double>::infinity();
                double max_rtt = 0.0;
                int iters = 0;

                MPI_Barrier(region_comm);
                double t0 = MPI_Wtime();

                // --- timed reduce ---
                for(int i = 0; i < num_iterations; i++)
                {
#if defined(USE_CUDA)
                    cuda_check(cudaMemcpy(rd_h_send, rd_d_send, red_count, cudaMemcpyDeviceToHost));
                    MPI_Reduce(rd_h_send, rd_h_recv, (int)red_count, MPI_CHAR, MPI_SUM, 0, region_comm);
                    cuda_check(cudaMemcpy(rd_d_recv, rd_h_recv, red_count, cudaMemcpyHostToDevice));
#elif defined(USE_HIP)
                    // hipMemcpy(rd_h_send, rd_d_send, red_count, hipMemcpyDeviceToHost);
                    // MPI_Reduce(rd_h_send, rd_h_recv, (int)red_count, MPI_CHAR, MPI_SUM, 0, MPI_COMM_WORLD);
                    // hipMemcpy(rd_d_recv, rd_h_recv, red_count, hipMemcpyHostToDevice);
                    MPI_Reduce(rd_d_send, rd_d_recv, (int)red_count, MPI_CHAR, MPI_SUM, 0, region_comm);
#else
                    MPI_Reduce(rd_send, rd_recv, (int)red_count, MPI_CHAR, MPI_SUM, 0, region_comm);
#endif
                }
                double dt = MPI_Wtime() - t0;
                double iter_max = 0.0;
                MPI_Reduce(&dt, &iter_max, 1, MPI_DOUBLE, MPI_MAX, 0, region_comm);

                if(region_rank == 0)
                {
                        // red_total_time += iter_max;
                        // if(iter_max < min_rtt) min_rtt = iter_max;      //fix min and max calculation
                        // if(iter_max > max_rtt) max_rtt = iter_max;
                        // ++iters;

                    double avg_rtt = iter_max / num_iterations;
#if defined(USE_CALIPER)
                    cali_set_string(comm_phase_attr, "reduce");
                    cali_set_double(red_avg_time_sec_attr, avg_rtt);
                    cali_set_double(red_max_time_sec_attr, iter_max);
                        //cali_set_double(red_min_time_sec_attr, min_rtt);
                    printf("REDUCE %s: total=%g s, avg_per_iter=%g s over %d iterations\n", region_label.c_str(), iter_max, avg_rtt, num_iterations);
                    fflush(stdout);
#endif
                }
                
#if defined(USE_CALIPER)
                CALI_MARK_END(region_label.c_str());
#endif
                printf("rank %d: end timing\n", rank);
                fflush(stdout);

#if defined(USE_HIP)
                free(rd_h_send);
                free(rd_h_recv);
                hipFree(rd_d_send);
                hipFree(rd_d_recv);
#elif defined(USE_CUDA)
                cuda_check(cudaFreeHost(rd_h_send));
                cuda_check(cudaFreeHost(rd_h_recv));
                cuda_check(cudaFree(rd_d_send));
                cuda_check(cudaFree(rd_d_recv));
#else
                free(rd_send);
                free(rd_recv);
#endif
                printf("freed memory\n");
                fflush(stdout);
                MPI_Comm_free(&region_comm);
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // ===================== ALLREDUCE =====================
        if (op == OpKind::Allreduce || op == OpKind::All)
        {
            for (int partner_rank : partners)
            {
                std::string region_label = region_names[partner_rank];

                int ranks_in_region = partner_rank + 1;
                int active = (rank < ranks_in_region);

                MPI_Comm region_comm = MPI_COMM_NULL;
                MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, rank, &region_comm);     //making subcommunicators

                if (!active)
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                    continue;
                }

                int region_rank = 0;
                int region_size = 0;
                MPI_Comm_rank(region_comm, &region_rank);
                MPI_Comm_size(region_comm, &region_size);

                size_t ar_count = static_cast<size_t>(message);
                if (ar_count > INT_MAX) {
                    if (rank == 0) fprintf(stderr, "Allreduce count too large\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                int warmup = 1;
                double ar_total_time = 0.0;

                if (region_rank == 0) {
                    printf("\n--- Testing %s (ALLREDUCE) with %d ranks ---\n", region_label.c_str(), region_size);
                    fflush(stdout);
                }

#if defined(USE_CUDA)
                char *ar_d_send=nullptr;
                char *ar_d_recv=nullptr;
                cuda_check(cudaMalloc((void**)&ar_d_send, ar_count));
                cuda_check(cudaMalloc((void**)&ar_d_recv, ar_count));

                char *ar_h_send=nullptr;
                char *ar_h_recv=nullptr;
                cuda_check(cudaMallocHost((void**)&ar_h_send, ar_count));
                cuda_check(cudaMallocHost((void**)&ar_h_recv, ar_count));

                fill_with_random_pattern(ar_h_send, ar_count);
                memset(ar_h_recv, 0,   ar_count);

                cuda_check(cudaMemcpy(ar_d_send, ar_h_send, ar_count, cudaMemcpyHostToDevice));
                cuda_check(cudaMemset(ar_d_recv, 0,   ar_count));

#elif defined(USE_HIP)
                char *ar_d_send=nullptr;
                char *ar_d_recv=nullptr;

                hipError_t a_err1 = hipMalloc((void**)&ar_d_send, ar_count);
                hipError_t a_err2 = hipMalloc((void**)&ar_d_recv, ar_count);

                if (a_err1 != hipSuccess || a_err2 != hipSuccess) {
                    fprintf(stderr, "HIP malloc failed: %s %s\n", hipGetErrorString(a_err1), hipGetErrorString(a_err2));
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                char *ar_h_send = (char*)malloc(ar_count);
                char *ar_h_recv = (char*)malloc(ar_count);
                fill_with_random_pattern(ar_h_send, ar_count);
                memset(ar_h_recv, 0,   ar_count);

                hipError_t a_cuerr1 = hipMemcpy(ar_d_send, ar_h_send, ar_count, hipMemcpyHostToDevice);
                assert(a_cuerr1 == hipSuccess);
                hipError_t a_cuerr2 = hipMemset(ar_d_recv, 0, ar_count);
                assert(a_cuerr2 == hipSuccess);
#else
                char *ar_send = (char*)malloc(ar_count);
                char *ar_recv = (char*)malloc(ar_count);
                fill_with_random_pattern(ar_send, ar_count);
                memset(ar_recv, 0,   ar_count);
#endif

#if defined(USE_CALIPER)
                CALI_MARK_BEGIN(warmup_region_ar);
#endif
                printf("rank %d: begin warmup\n", rank);
                fflush(stdout);

                for (int i = 0; i < warmup; i++)
                {
#if defined(USE_CUDA)
                    cuda_check(cudaMemcpy(ar_h_send, ar_d_send, ar_count, cudaMemcpyDeviceToHost));
                    MPI_Allreduce(ar_h_send, ar_h_recv, (int)ar_count, MPI_CHAR, MPI_SUM, region_comm);
                    cuda_check(cudaMemcpy(ar_d_recv, ar_h_recv, ar_count, cudaMemcpyHostToDevice));
#elif defined(USE_HIP)
                    // hipMemcpy(ar_h_send, ar_d_send, ar_count, hipMemcpyDeviceToHost);
                    // MPI_Allreduce(ar_h_send, ar_h_recv, (int)ar_count, MPI_CHAR, MPI_SUM, region_comm);
                    // hipMemcpy(ar_d_recv, ar_h_recv, ar_count, hipMemcpyHostToDevice);
                    MPI_Allreduce(ar_d_send, ar_d_recv, (int)ar_count, MPI_CHAR, MPI_SUM, region_comm);
#else
                    MPI_Allreduce(ar_send, ar_recv, (int)ar_count, MPI_CHAR, MPI_SUM, region_comm);
#endif
                }
#if defined(USE_CALIPER)
                CALI_MARK_END(warmup_region_ar);
                CALI_MARK_BEGIN(region_label.c_str());
#endif
                printf("rank %d: end warmup, begin timing\n", rank);
                fflush(stdout);

                double min_rtt = std::numeric_limits<double>::infinity();
                double max_rtt = 0.0;
                int iters = 0;

                // --- timed allreduce: back-to-back MPI calls ---
                MPI_Barrier(region_comm);

                double t0 = MPI_Wtime();

                for (int it = 0; it < num_iterations; ++it)
                {
#if defined(USE_CUDA)
                    cuda_check(cudaMemcpy(ar_h_send, ar_d_send, ar_count, cudaMemcpyDeviceToHost));
                    MPI_Allreduce(ar_h_send, ar_h_recv, (int)ar_count, MPI_CHAR, MPI_SUM, region_comm);
                    cuda_check(cudaMemcpy(ar_d_recv, ar_h_recv, ar_count, cudaMemcpyHostToDevice));
#elif defined(USE_HIP)
                    // hipMemcpy(ar_h_send, ar_d_send, ar_count, hipMemcpyDeviceToHost);
                    // MPI_Allreduce(ar_h_send, ar_h_recv, (int)ar_count, MPI_CHAR, MPI_SUM, region_comm);
                    // hipMemcpy(ar_d_recv, ar_h_recv, ar_count, hipMemcpyHostToDevice);
                    MPI_Allreduce(ar_d_send, ar_d_recv, (int)ar_count, MPI_CHAR, MPI_SUM, region_comm);
#else
                    MPI_Allreduce(ar_send, ar_recv, (int)ar_count, MPI_CHAR, MPI_SUM,region_comm);
#endif
                }

                double local_total_time = MPI_Wtime() - t0;
                double max_total_time = 0.0;
                MPI_Allreduce(&local_total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, region_comm);

                if (region_rank == 0)
                {
                    double avg_time = max_total_time / num_iterations;
#if defined(USE_CALIPER)
                    cali_set_string(comm_phase_attr, "allreduce");
                    cali_set_double(ar_avg_time_sec_attr, avg_time);
                    cali_set_double(ar_max_time_sec_attr, max_total_time);      //max total time for # iterations
                    //cali_set_double(ar_min_time_sec_attr, avg_time);
#endif
                    printf("ALLREDUCE %s: total=%g s, avg_per_iter=%g s over %d iterations\n",
                        region_label.c_str(), max_total_time, avg_time, num_iterations);
                    fflush(stdout);
                }
#if defined(USE_CALIPER)
                CALI_MARK_END(region_label.c_str());
#endif
                printf("rank %d: end timing\n", rank);
                fflush(stdout);

#if defined(USE_HIP)
                free(ar_h_send);
                free(ar_h_recv);
                hipFree(ar_d_send);
                hipFree(ar_d_recv);
#elif defined(USE_CUDA)
                cuda_check(cudaFreeHost(ar_h_send));
                cuda_check(cudaFreeHost(ar_h_recv));
                cuda_check(cudaFree(ar_d_send));
                cuda_check(cudaFree(ar_d_recv));
#else
                free(ar_send);
                free(ar_recv);
#endif
                printf("freed memory\n");
                fflush(stdout);

                MPI_Comm_free(&region_comm);
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // ===================== BANDWIDTH =====================
        if (op == OpKind::Bandwidth || op == OpKind::All)
        {
            for (int partner_rank : partners)
            {
                std::string region_label = region_names[partner_rank];

                std::vector<RankPair> pairs =
                    build_pingpong_pairs(region_label,
                                        size,
                                        sys_cores_per_socket,
                                        sys_cores_per_node,
                                        pingpong_num_pairs);

                if (pairs.empty()) {
                    if (rank == 0)
                        printf("Skipping region %s: no valid bandwidth pairs\n", region_label.c_str());
                    continue;
                }

                std::map<int, int> src_to_dst;
                std::map<int, int> dst_to_src;
                std::set<int> active_ranks;

                for (auto &p : pairs) {
                    src_to_dst[p.src] = p.dst;
                    dst_to_src[p.dst] = p.src;
                    active_ranks.insert(p.src);
                    active_ranks.insert(p.dst);
                }

                int active = active_ranks.count(rank) > 0;

                MPI_Comm bw_comm = MPI_COMM_NULL;
                MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, rank, &bw_comm);

                if (!active) {
                    MPI_Barrier(MPI_COMM_WORLD);
                    continue;
                }

                int bw_rank = 0;
                MPI_Comm_rank(bw_comm, &bw_rank);

                bool is_sender = src_to_dst.count(rank) > 0;
                bool is_receiver = dst_to_src.count(rank) > 0;

                int peer = MPI_PROC_NULL;
                if (is_sender) {
                    peer = src_to_dst[rank];
                }
                else if (is_receiver) {
                    peer = dst_to_src[rank];
                }

                if (rank == 0)
                {
                    printf("\n--- Testing %s (BANDWIDTH) with %zu pairs ---\n", region_label.c_str(), pairs.size());

                    for (auto &p : pairs) {
                        printf("  one-way pair %d (%s) -> %d (%s)\n", p.src, all_hostnames[p.src], p.dst, all_hostnames[p.dst]);
                    }
                    fflush(stdout);

#if defined(USE_CALIPER)
                    cali_set_int(src_rank_attr, pairs[0].src);
                    cali_set_int(dest_rank_attr, pairs[0].dst);
                    cali_set_int(src_node_attr, extract_node_number(all_hostnames[pairs[0].src]));
                    cali_set_int(dest_node_attr, extract_node_number(all_hostnames[pairs[0].dst]));
#endif
                }

                const int base_tag = 3000 + ((partner_rank % 2000) * 10);
                const int BW_TAG = base_tag;

                int warmup = 1;
                size_t bw_total_bytes = (size_t)WINDOW_SIZE * (size_t)message;

                // ---------- buffer allocation ----------
#if defined(USE_HIP)
                char *send_flat = nullptr;
                char *recv_flat = nullptr;

                hipError_t err1 = hipMalloc((void**)&send_flat, bw_total_bytes);
                hipError_t err2 = hipMalloc((void**)&recv_flat, bw_total_bytes);

                if (err1 != hipSuccess || err2 != hipSuccess) {
                    fprintf(stderr, "HIP malloc failed: %s %s\n", hipGetErrorString(err1), hipGetErrorString(err2));
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                char *h_tmp = (char*)malloc(bw_total_bytes);
                if (!h_tmp) {
                    fprintf(stderr, "Rank %d host malloc failed for %zu bytes\n", rank, bw_total_bytes);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                fill_with_random_pattern(h_tmp, bw_total_bytes);

                hipError_t cuerr1 = hipMemcpy(send_flat, h_tmp, bw_total_bytes, hipMemcpyHostToDevice);
                assert(cuerr1 == hipSuccess);
                hipError_t cuerr2 = hipMemset(recv_flat, 0, bw_total_bytes);
                assert(cuerr2 == hipSuccess);

                free(h_tmp);

#elif defined(USE_CUDA)
                int dev_count = 0;
                cuda_check(cudaGetDeviceCount(&dev_count));
                cuda_check(cudaSetDevice(rank % (dev_count > 0 ? dev_count : 1)));

                char *send_flat = nullptr;
                char *recv_flat = nullptr;

                cuda_check(cudaMalloc((void**)&send_flat, bw_total_bytes));
                cuda_check(cudaMalloc((void**)&recv_flat, bw_total_bytes));

                char *h_tmp = nullptr;
                cuda_check(cudaMallocHost((void**)&h_tmp, bw_total_bytes));
                fill_with_random_pattern(h_tmp, bw_total_bytes);
                cuda_check(cudaMemcpy(send_flat, h_tmp, bw_total_bytes, cudaMemcpyHostToDevice));
                cuda_check(cudaMemset(recv_flat, 0, bw_total_bytes));
                cuda_check(cudaFreeHost(h_tmp));

#else
                char *send_flat = (char*)malloc(bw_total_bytes);
                char *recv_flat = (char*)malloc(bw_total_bytes);

                if (!send_flat || !recv_flat) {
                    fprintf(stderr, "Rank %d malloc failed for %zu bytes\n", rank, bw_total_bytes);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                fill_with_random_pattern(send_flat, bw_total_bytes);
                memset(recv_flat, 0, bw_total_bytes);
#endif

                std::vector<char*> s_buf(WINDOW_SIZE);
                std::vector<char*> r_buf(WINDOW_SIZE);

                for (int j = 0; j < WINDOW_SIZE; ++j) {
                    s_buf[j] = send_flat + (size_t)j * (size_t)message;
                    r_buf[j] = recv_flat + (size_t)j * (size_t)message;
                }

                std::vector<MPI_Request> requests(WINDOW_SIZE);

                MPI_Barrier(bw_comm);

                // ---------- warmup ----------
                for (int i = 0; i < warmup; ++i)
                {
                    if (is_receiver) {
                        for (int j = 0; j < WINDOW_SIZE; ++j) {
                            MPI_Irecv(r_buf[j], message, MPI_CHAR, peer, BW_TAG + j, MPI_COMM_WORLD, &requests[j]);
                        }
                    }

                    MPI_Barrier(bw_comm);

                    if (is_sender) {
                        for (int j = 0; j < WINDOW_SIZE; ++j) {
                            MPI_Isend(s_buf[j], message, MPI_CHAR, peer, BW_TAG + j, MPI_COMM_WORLD, &requests[j]);
                        }
                        MPI_Waitall(WINDOW_SIZE, requests.data(), MPI_STATUSES_IGNORE);
                    }
                    if (is_receiver) {
                        MPI_Waitall(WINDOW_SIZE, requests.data(), MPI_STATUSES_IGNORE);
                    }

                    MPI_Barrier(bw_comm);
                }

#if defined(USE_CALIPER)
                CALI_MARK_BEGIN(region_label.c_str());
#endif

                // ---------- timed bandwidth ----------
                double total_time = 0.0;
                double min_time = std::numeric_limits<double>::infinity();
                double max_time = 0.0;
                int iters = 0;

                MPI_Barrier(bw_comm);

                for (int i = 0; i < num_iterations; ++i)
                {
                    if (is_receiver) {
                        for (int j = 0; j < WINDOW_SIZE; ++j) {
                            MPI_Irecv(r_buf[j], message, MPI_CHAR, peer, BW_TAG + j, MPI_COMM_WORLD, &requests[j]);
                        }
                    }

                    MPI_Barrier(bw_comm);

                    if (is_sender) {
                        double start = MPI_Wtime();
                        for (int j = 0; j < WINDOW_SIZE; ++j) {
                            MPI_Isend(s_buf[j], message, MPI_CHAR, peer, BW_TAG + j, MPI_COMM_WORLD, &requests[j]);
                        }

                        MPI_Waitall(WINDOW_SIZE, requests.data(), MPI_STATUSES_IGNORE);

                        double end = MPI_Wtime();
                        double dt = end - start;

                        total_time += dt;
                        if (dt < min_time) min_time = dt;
                        if (dt > max_time) max_time = dt;
                        ++iters;
                    }

                    if (is_receiver) {
                        MPI_Waitall(WINDOW_SIZE, requests.data(), MPI_STATUSES_IGNORE);
                    }

                    MPI_Barrier(bw_comm);
                }

                double local_avg = 0.0;
                double local_min = std::numeric_limits<double>::infinity();
                double local_max = 0.0;

                if (is_sender && iters > 0) {
                    local_avg = total_time / iters;
                    local_min = min_time;
                    local_max = max_time;
                }

                double bandwidth_avg_time = 0.0;
                double bandwidth_min_time = 0.0;
                double bandwidth_max_time = 0.0;

                MPI_Reduce(&local_avg, &bandwidth_avg_time, 1, MPI_DOUBLE, MPI_MAX, 0, bw_comm);
                MPI_Reduce(&local_min, &bandwidth_min_time, 1, MPI_DOUBLE, MPI_MIN, 0, bw_comm);
                MPI_Reduce(&local_max, &bandwidth_max_time, 1, MPI_DOUBLE, MPI_MAX, 0, bw_comm);

                if (bw_rank == 0)
                {
#if defined(USE_CALIPER)
                    cali_set_string(comm_phase_attr, "bandwidth");
                    cali_set_double(bw_avg_time_sec_attr, bandwidth_avg_time);
                    cali_set_double(bw_min_time_sec_attr, bandwidth_min_time);
                    cali_set_double(bw_max_time_sec_attr, bandwidth_max_time);
#endif

                    printf("BANDWIDTH %s: avg=%g s, min=%g s, max=%g s\n", region_label.c_str(), bandwidth_avg_time, bandwidth_min_time, bandwidth_max_time);
                    fflush(stdout);
                }

#if defined(USE_CALIPER)
                CALI_MARK_END(region_label.c_str());
#endif

                // ---------- cleanup ----------
#if defined(USE_HIP)
                hipFree(send_flat);
                hipFree(recv_flat);
#elif defined(USE_CUDA)
                cuda_check(cudaFree(send_flat));
                cuda_check(cudaFree(recv_flat));
#else
                free(send_flat);
                free(recv_flat);
#endif

                MPI_Comm_free(&bw_comm);
                MPI_Barrier(MPI_COMM_WORLD);
            }

            if (rank == 0)
                printf("Done with Bandwidth\n");
        }

#if defined(USE_CALIPER)
        mgr[message].stop();
#endif
    }

#if defined(USE_CALIPER)
    if (rank == 0 && !all_comm_pairs.empty())
    {
        adiak::value("all_comm_pairs", all_comm_pairs);
    }
    for (auto &m : mgr)
    {
        m.second.flush();
    }
#endif

    printf("rank %d\n", rank);
    MPI_Finalize();
    return 0;
}
