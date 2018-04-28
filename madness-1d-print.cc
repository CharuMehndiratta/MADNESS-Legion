#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath> // pow
#include "legion.h"

using namespace Legion;
using namespace std;

enum TASK_IDs {
    TOP_LEVEL_TASK_ID,
    REFINE_TASK_ID,
    SET_TASK_ID,
    PRINT_TASK_ID,
    READ_TASK_ID,
    COMPRESS_TASK_ID,
    COMPRESS_SET_TASK_ID,
    GET_COEF_TASK_ID,
    DIFF_TASK_ID,
    DIFF_SET_TASK_ID
};

enum FieldIDs {
    FID_X,
};

struct Arguments {
    /* level of the node in the binary tree. Root is at level 0 */
    int n;

    /* labeling of the node in the binary tree. Root has the value label = 0 
    * Node with (n, l) has it's left child at (n + 1, 2 * l) and it's right child at (n + 1, 2 * l + 1)
    */
    int l;

    int max_depth;

    coord_t idx;

    drand48_data gen;

    Color partition_color;

    int actual_max_depth;

    Arguments(int _n, int _l, int _max_depth, coord_t _idx, Color _partition_color, int _actual_max_depth=0)
        : n(_n), l(_l), max_depth(_max_depth), idx(_idx), partition_color(_partition_color), actual_max_depth(_actual_max_depth)
    {
        if (_actual_max_depth == 0) {
            actual_max_depth = _max_depth;
        }
    }
};

struct SetTaskArgs {
    int node_value;
    coord_t idx;
    int n;
    int max_depth;
    SetTaskArgs(int _node_value, coord_t _idx, int _n, int _max_depth) : node_value(_node_value), idx(_idx), n(_n), max_depth(_max_depth) {}
};

struct ReadTaskArgs {
    coord_t idx;
    ReadTaskArgs(coord_t _idx) : idx(_idx) {}
};

struct CompressSetTaskArgs {
    coord_t idx, left_idx, right_idx;
    CompressSetTaskArgs(coord_t _idx, coord_t _left_idx, coord_t _right_idx) : 
        idx(_idx), left_idx(_left_idx), right_idx(_right_idx){}
};


struct DiffArguments {
    /* level of the node in the binary tree. Root is at level 0 */
    int n, l, max_depth;
    coord_t idx;
    Color partition_color1;
    Color partition_color2;
    int actual_max_depth;
    int s0;
    bool is_s0_valid;

    DiffArguments(int _n, int _l, int _max_depth, coord_t _idx, Color _partition_color1, Color _partition_color2, int _actual_max_depth, int _s0, bool _is_s0_valid)
        : n(_n), l(_l), max_depth(_max_depth), idx(_idx), partition_color1(_partition_color1), partition_color2(_partition_color2),
        s0(_s0), is_s0_valid(_is_s0_valid), actual_max_depth(_actual_max_depth)
    {}
};

struct DiffSetTaskArgs {
    coord_t idx;
    int node_value;
    DiffSetTaskArgs(coord_t _idx, int _node_value) : 
        idx(_idx), node_value(_node_value) {}
};

struct GetCoefArguments {
    /* level of the node in the binary tree. Root is at level 0 */
    int n, l, max_depth;
    coord_t idx;
    Color partition_color;
    int questioned_n, questioned_l;

    GetCoefArguments(int _n, int _l, int _max_depth, coord_t _idx, Color _partition_color, int _questioned_n, int _questioned_l)
        : n(_n), l(_l), max_depth(_max_depth), idx(_idx), partition_color(_partition_color),
        questioned_n(_questioned_n), questioned_l(_questioned_l)

    {}
};

//   k=1 (1 subregion per node)
//                0
//         1             8
//     2      5      9      12
//   3   4  6   7  10  11  13   14
//
//       i              (n, l)
//    il    ir   (n+1, 2*l)  (n+1, 2*l+1)
//
//    il = i + 1
//    ir = i + 2^(max_level -l)
//
//    when each subtree holds k levels
//    [i .. i+(2^k-1)-1]
//    0 <= j <= 2^k-1 => [i+(2^k-1)-1 + 1 +  j      * (2^(max_level - (l + k) +1) - 1) ..
//                        i+(2^k-1)-1 + 1 + (j + 1) * (2^(max_level - (l + k) +1) - 1) - 1]
void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {

    int overall_max_depth = 4;
    int actual_left_depth = 4;
    long int seed = 12345;
    {
        const InputArgs &command_args = HighLevelRuntime::get_input_args();
        for (int idx = 1; idx < command_args.argc; ++idx)
        {
            if (strcmp(command_args.argv[idx], "-max_depth") == 0)
                overall_max_depth = atoi(command_args.argv[++idx]);
            else if (strcmp(command_args.argv[idx], "-seed") == 0)
                seed = atol(command_args.argv[++idx]);
        }
    }

    Rect<1> tree_rect(0LL, static_cast<coord_t>(pow(2, overall_max_depth + 1)) - 2);
    IndexSpace is = runtime->create_index_space(ctx, tree_rect);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(int), FID_X);
    }

    // For 1 logical region
    LogicalRegion lr1 = runtime->create_logical_region(ctx, is, fs);
    // Any random value will work
    Color partition_color1 = 10;

    Arguments args(0, 0, overall_max_depth, 0, partition_color1, actual_left_depth);
    srand48_r(seed, &args.gen);

    // Launching the refine task
    TaskLauncher refine_launcher(REFINE_TASK_ID, TaskArgument(&args, sizeof(Arguments)));
    refine_launcher.add_region_requirement(RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1));
    refine_launcher.add_field(0, FID_X);
    runtime->execute_task(ctx, refine_launcher);

    // // Launching another task to print the values of the binary tree nodes
    // TaskLauncher print_launcher(PRINT_TASK_ID, TaskArgument(&args, sizeof(Arguments)));
    // print_launcher.add_region_requirement(RegionRequirement(lr1, READ_ONLY, EXCLUSIVE, lr1));
    // print_launcher.add_field(0, FID_X);
    // runtime->execute_task(ctx, print_launcher);

    // Launching another task to print the values of the binary tree nodes
    TaskLauncher compress_launcher(COMPRESS_TASK_ID, TaskArgument(&args, sizeof(Arguments)));
    compress_launcher.add_region_requirement(RegionRequirement(lr1, READ_WRITE, EXCLUSIVE, lr1));
    compress_launcher.add_field(0, FID_X);
    runtime->execute_task(ctx, compress_launcher);

    // Launching another task to print the values of the binary tree nodes
    TaskLauncher print_launcher1(PRINT_TASK_ID, TaskArgument(&args, sizeof(Arguments)));
    print_launcher1.add_region_requirement(RegionRequirement(lr1, READ_ONLY, EXCLUSIVE, lr1));
    print_launcher1.add_field(0, FID_X);
    runtime->execute_task(ctx, print_launcher1);

    // GetCoefArguments get_coef_args(0, 0, overall_max_depth, 0, partition_color1, 2, 0);

    // Future f1;
    // {
    //     TaskLauncher get_coefs_launcher(GET_COEF_TASK_ID, TaskArgument(&get_coef_args, sizeof(GetCoefArguments)));
    //     get_coefs_launcher.add_region_requirement(RegionRequirement(lr1, READ_ONLY, EXCLUSIVE, lr1));
    //     get_coefs_launcher.add_field(0, FID_X);
    //     f1 = runtime->execute_task(ctx, get_coefs_launcher);
    // }

    // fprintf(stderr, "get coefs %d\n", f1.get_result<int>());

    Color partition_color2 = 50;
    DiffArguments diff_args(0, 0, overall_max_depth, 0, partition_color1, partition_color2, actual_left_depth, 100, false);

    Rect<1> tree_rect2(0LL, static_cast<coord_t>(pow(2, overall_max_depth + 1)) - 2);
    IndexSpace is2 = runtime->create_index_space(ctx, tree_rect2);
    LogicalRegion lr2 = runtime->create_logical_region(ctx, is2, fs);

    TaskLauncher diff_launcher(DIFF_TASK_ID, TaskArgument(&diff_args, sizeof(DiffArguments)));
    diff_launcher.add_region_requirement(RegionRequirement(lr1, READ_ONLY, EXCLUSIVE, lr1));
    diff_launcher.add_region_requirement(RegionRequirement(lr2, WRITE_DISCARD, EXCLUSIVE, lr2));
    diff_launcher.add_region_requirement(RegionRequirement(lr1, READ_ONLY, EXCLUSIVE, lr1));
    diff_launcher.add_field(0, FID_X);
    diff_launcher.add_field(1, FID_X);
    diff_launcher.add_field(2, FID_X);
    runtime->execute_task(ctx, diff_launcher);

    Arguments args2(0, 0, overall_max_depth, 0, partition_color2, actual_left_depth);

    // Launching another task to print the values of the binary tree nodes
    TaskLauncher print_launcher12(PRINT_TASK_ID, TaskArgument(&args2, sizeof(Arguments)));
    print_launcher12.add_region_requirement(RegionRequirement(lr2, READ_ONLY, EXCLUSIVE, lr2));
    print_launcher12.add_field(0, FID_X);
    runtime->execute_task(ctx, print_launcher12);

    // Destroying allocated memory
    runtime->destroy_logical_region(ctx, lr1);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
}


/*
 *
 *  This algorithm generates a binary tree (and only leaves contain the valuable data). Initial call would be Refine(0,0):
 *   
 *  1) Refine(int n, int l) {
 *  2)        int node_value = pick a random value in a range [1, 10], inclusive;
 *
 *  3)        if (node_value <= 3 || n >= MAX_DEPTH) {
 *  4)                   store in the hash_map (n, l) --> node_value;
 *  5)        }
 *  6)        else {
 *  7)                   store in the hash_map (n, l) --> ZERO; // ZERO value indicates that the node is an internal node
 *  8)                   make a new task of Refine with arguments(n+1, 2 * l); // left child
 *  9)                   make a new task of Refine with arguments(n+1, 2 * l + 1); // right child
 *  10)      }
 *  11) }
 *
 *  
 *
 *  So, as you can clearly see, the result of this task is a binary tree, whose internal nodes contain the value ZERO and only it's leaves contain values in range [1, 3]. As an example, the following could be the result of running the ALG-1:
 *
 *                        _____________0_____________                                 DEPTH/LEVEL = 0
 *                  _____0____                 ______0_______                         DEPTH/LEVEL = 1
 *             ____0___       1            ___0___         __0____                    DEPTH/LEVEL = 2
 *            2        1                  3     __0__     1     __0__                 DEPTH/LEVEL = 3
 *                                           __0__   3         1     2                DEPTH/LEVEL = 4
 *                                          2     2                                   DEPTH/LEVEL = 5
 *
 *
 *  This tree is called to be in "scaling or refined form".
 *
 * */


void set_task(const Task *task,
              const std::vector<PhysicalRegion> &regions,
              Context ctx, HighLevelRuntime *runtime) {

    SetTaskArgs args = *(const SetTaskArgs *) task->args;
    assert(regions.size() == 1);
    const FieldAccessor<WRITE_DISCARD, int, 1> write_acc(regions[0], FID_X);
    if (args.node_value <= 3 || args.n == args.max_depth - 1) {
        write_acc[args.idx] = args.node_value % 3 + 1;
    }
    else {
        write_acc[args.idx] = 0;
    }
}

int read_task(const Task *task,
              const std::vector<PhysicalRegion> &regions,
              Context ctx, HighLevelRuntime *runtime) {

    ReadTaskArgs args = *(const ReadTaskArgs *) task->args;
    assert(regions.size() == 1);
    const FieldAccessor<READ_ONLY, int, 1> read_acc(regions[0], FID_X);
    return read_acc[args.idx];
}

void compress_set_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, HighLevelRuntime *runtime) {

    CompressSetTaskArgs args = *(const CompressSetTaskArgs *) task->args;
    assert(regions.size() == 3);
    const FieldAccessor<READ_WRITE, int, 1> write_acc(regions[0], FID_X);
    const FieldAccessor<READ_WRITE, int, 1> write_acc_left(regions[1], FID_X);
    const FieldAccessor<READ_WRITE, int, 1> write_acc_right(regions[2], FID_X);

    write_acc[args.idx] = write_acc_left[args.left_idx] + write_acc_right[args.right_idx];
}

void diff_set_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {

    DiffSetTaskArgs args = *(const DiffSetTaskArgs *) task->args;
    assert(regions.size() == 1);

    LogicalRegion lr = regions[0].get_logical_region();
    assert(lr != LogicalRegion::NO_REGION);

    const FieldAccessor<READ_WRITE, int, 1> write_acc(regions[0], FID_X);

    write_acc[args.idx] = args.node_value;
    // fprintf(stderr, "step 6\n");
}


// To be recursive task calling for the left and right subtrees, if necessary !
void refine_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {

    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    int n = args.n;
    int l = args.l;
    int max_depth = args.max_depth;
    int actual_max_depth = args.actual_max_depth;

    DomainPoint my_sub_tree_color(Point<1>(0LL));
    DomainPoint left_sub_tree_color(Point<1>(1LL));
    DomainPoint right_sub_tree_color(Point<1>(2LL));
    Color partition_color = args.partition_color;

    coord_t idx = args.idx;

    assert(regions.size() == 1);
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalPartition lp = LogicalPartition::NO_PART;
    LogicalRegion my_sub_tree_lr = lr;

    coord_t idx_left_sub_tree = 0LL;
    coord_t idx_right_sub_tree = 0LL;


    if (n < actual_max_depth)
    {
        IndexSpace is = lr.get_index_space();
        DomainPointColoring coloring;

        idx_left_sub_tree = idx + 1;
        idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));

        Rect<1> my_sub_tree_rect(idx, idx);
        Rect<1> left_sub_tree_rect(idx_left_sub_tree, idx_right_sub_tree - 1);
        Rect<1> right_sub_tree_rect(idx_right_sub_tree,
        idx_right_sub_tree + static_cast<coord_t>(pow(2, max_depth - n)) - 2);
        /*
        fprintf(stderr, "(n: %d, l: %d) - idx: [%lld, %lld] (max_depth: %d)\n"
        "  |-- (n: %d, l: %d) - idx: [%lld, %lld] (max_depth: %d)\n"
        "  |-- (n: %d, l: %d) - idx: [%lld, %lld] (max_depth: %d)\n",
        n, l, idx, idx, max_depth,
        n + 1, 2 * l,     left_sub_tree_rect.lo[0],  left_sub_tree_rect.hi[0],  max_depth,
        n + 1, 2 * l + 1, right_sub_tree_rect.lo[0], right_sub_tree_rect.hi[0], max_depth); */

        coloring[my_sub_tree_color] = my_sub_tree_rect;
        coloring[left_sub_tree_color] = left_sub_tree_rect;
        coloring[right_sub_tree_color] = right_sub_tree_rect;

        Rect<1> color_space = Rect<1>(my_sub_tree_color, right_sub_tree_color);

        IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, coloring, DISJOINT_KIND, partition_color);
        lp = runtime->get_logical_partition(ctx, lr, ip);
        my_sub_tree_lr = runtime->get_logical_subregion_by_color(ctx, lp, my_sub_tree_color);
    }

    assert(lr != LogicalRegion::NO_REGION);
    assert(my_sub_tree_lr != LogicalRegion::NO_REGION);

    long int node_value;
    lrand48_r(&args.gen, &node_value);
    node_value = node_value % 10 + 1;
    {
        SetTaskArgs args(node_value, idx, n, actual_max_depth);
        TaskLauncher set_task_launcher(SET_TASK_ID, TaskArgument(&args, sizeof(SetTaskArgs)));
        RegionRequirement req(my_sub_tree_lr, WRITE_DISCARD, EXCLUSIVE, lr);
        req.add_field(FID_X);
        set_task_launcher.add_region_requirement(req);
        runtime->execute_task(ctx, set_task_launcher);
    }

    if (node_value > 3 && n < actual_max_depth)
    {
        assert(lp != LogicalPartition::NO_PART);
        Rect<1> launch_domain(left_sub_tree_color, right_sub_tree_color);
        ArgumentMap arg_map;

        Arguments for_left_sub_tree (n + 1, l * 2    , max_depth, idx_left_sub_tree, partition_color, actual_max_depth);
        Arguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, partition_color, actual_max_depth);

        // Make sure two subtrees use different random number generators
        long int new_seed = 0L;
        lrand48_r(&args.gen, &new_seed);
        for_left_sub_tree.gen = args.gen;
        srand48_r(new_seed, &for_right_sub_tree.gen);

        arg_map.set_point(left_sub_tree_color, TaskArgument(&for_left_sub_tree, sizeof(Arguments)));
        arg_map.set_point(right_sub_tree_color, TaskArgument(&for_right_sub_tree, sizeof(Arguments)));

        IndexTaskLauncher refine_launcher(REFINE_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        RegionRequirement req(lp, 0, WRITE_DISCARD, EXCLUSIVE, lr);
        req.add_field(FID_X);
        refine_launcher.add_region_requirement(req);
        runtime->execute_index_space(ctx, refine_launcher);
    }
}

void compress_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctxt, HighLevelRuntime *runtime) {
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;

    int n = args.n;
    int l = args.l;
    int max_depth = args.max_depth;

    DomainPoint my_sub_tree_color(Point<1>(0LL));
    DomainPoint left_sub_tree_color(Point<1>(1LL));
    DomainPoint right_sub_tree_color(Point<1>(2LL));
    Color partition_color = args.partition_color;

    coord_t idx = args.idx;

    assert(regions.size() == 1);
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalPartition lp = LogicalPartition::NO_PART, lp1,lp2;

    coord_t idx_left_sub_tree = 0LL;
    coord_t idx_right_sub_tree = 0LL;

    lp = runtime->get_logical_partition_by_color(ctxt, lr, partition_color);
    LogicalRegion my_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, my_sub_tree_color);
    LogicalRegion left_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, left_sub_tree_color);
    LogicalRegion right_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, right_sub_tree_color);

    IndexSpace indexspace_left = left_sub_tree_lr.get_index_space();

    if (runtime->has_index_partition(ctxt, indexspace_left, partition_color)) {

        idx_left_sub_tree = idx + 1;
        idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));

        Rect<1> launch_domain(left_sub_tree_color, right_sub_tree_color);
        ArgumentMap arg_map;

        Arguments for_left_sub_tree(n + 1, 2 * l, max_depth, idx_left_sub_tree, partition_color);
        Arguments for_right_sub_tree(n + 1, 2 * l + 1, max_depth, idx_right_sub_tree, partition_color);

        arg_map.set_point(left_sub_tree_color, TaskArgument(&for_left_sub_tree, sizeof(Arguments)));
        arg_map.set_point(right_sub_tree_color, TaskArgument(&for_right_sub_tree, sizeof(Arguments)));

        IndexTaskLauncher compress_launcher(COMPRESS_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        RegionRequirement req(lp, 0, READ_WRITE, EXCLUSIVE, lr);
        req.add_field(FID_X);
        compress_launcher.add_region_requirement(req);
        runtime->execute_index_space(ctxt, compress_launcher);

        {
            CompressSetTaskArgs args(idx, idx_left_sub_tree, idx_right_sub_tree);
            TaskLauncher compress_set_task_launcher(COMPRESS_SET_TASK_ID, TaskArgument(&args, sizeof(CompressSetTaskArgs)));
            RegionRequirement req(my_sub_tree_lr, READ_WRITE, EXCLUSIVE, lr);
            RegionRequirement req_left(left_sub_tree_lr, READ_WRITE, EXCLUSIVE, lr);
            RegionRequirement req_right(right_sub_tree_lr, READ_WRITE, EXCLUSIVE, lr);
            req.add_field(FID_X);
            req_left.add_field(FID_X);
            req_right.add_field(FID_X);
            compress_set_task_launcher.add_region_requirement(req);
            compress_set_task_launcher.add_region_requirement(req_left);
            compress_set_task_launcher.add_region_requirement(req_right);
            runtime->execute_task(ctxt, compress_set_task_launcher);
        }
    }
}

int get_coef_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctxt, HighLevelRuntime *runtime) {
    GetCoefArguments args = task->is_index_space ? *(const GetCoefArguments *) task->local_args
    : *(const GetCoefArguments *) task->args;

    int n = args.n;
    int l = args.l;
    int max_depth = args.max_depth;
    int questioned_n = args.questioned_n;
    int questioned_l = args.questioned_l;

    if(l < 0 || l >= pow(2, n)) {
        return 0;
    }

    DomainPoint my_sub_tree_color(Point<1>(0LL));
    DomainPoint left_sub_tree_color(Point<1>(1LL));
    DomainPoint right_sub_tree_color(Point<1>(2LL));
    Color partition_color = args.partition_color;

    // fprintf(stderr, "new partition_color1 %d\n", partition_color);

    coord_t idx = args.idx;

    assert(regions.size() == 1);
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalPartition lp = LogicalPartition::NO_PART;

    coord_t idx_left_sub_tree = 0LL;
    coord_t idx_right_sub_tree = 0LL;

    // fprintf(stderr, "I am in get_coef task2\n");

    lp = runtime->get_logical_partition_by_color(ctxt, lr, partition_color);

    // fprintf(stderr, "I am in get_coef task3\n");
    LogicalRegion my_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, my_sub_tree_color);
    LogicalRegion left_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, left_sub_tree_color);

    IndexSpace indexspace_left = left_sub_tree_lr.get_index_space();
    Future f1;
    FutureMap f2;

    // fprintf(stderr, "I am in get_coef task4\n");

    if (n == questioned_n && l == questioned_l) {
        {
            ReadTaskArgs args(idx);
            TaskLauncher read_task_launcher(READ_TASK_ID, TaskArgument(&args, sizeof(ReadTaskArgs)));
            RegionRequirement req(my_sub_tree_lr, READ_ONLY, EXCLUSIVE, lr);
            req.add_field(FID_X);
            read_task_launcher.add_region_requirement(req);
            f1 = runtime->execute_task(ctxt, read_task_launcher);
        }
        return f1.get_result<int>();
    }

    // fprintf(stderr, /"I am in get_coef task5\n");

    fprintf(stderr, "get coef n is %d l is %d\n", n, l);

    if (runtime->has_index_partition(ctxt, indexspace_left, partition_color)) {

        // fprintf(stderr, "I am in get_coef task6\n");
        idx_left_sub_tree = idx + 1;
        idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));

        Rect<1> launch_domain(left_sub_tree_color, right_sub_tree_color);
        ArgumentMap arg_map;

        GetCoefArguments for_left_sub_tree(n + 1, 2 * l, max_depth, idx_left_sub_tree, partition_color, questioned_n, questioned_l);
        GetCoefArguments for_right_sub_tree(n + 1, 2 * l + 1, max_depth, idx_right_sub_tree, partition_color, questioned_n, questioned_l);

        arg_map.set_point(left_sub_tree_color, TaskArgument(&for_left_sub_tree, sizeof(GetCoefArguments)));
        arg_map.set_point(right_sub_tree_color, TaskArgument(&for_right_sub_tree, sizeof(GetCoefArguments)));

        IndexTaskLauncher get_coefs_launcher(GET_COEF_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        RegionRequirement req(lp, 0, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(FID_X);
        get_coefs_launcher.add_region_requirement(req);
        f2 = runtime->execute_index_space(ctxt, get_coefs_launcher);
        f2.wait_all_results();
        if (f2.get_result<int>(left_sub_tree_color) != -1)
            return f2.get_result<int>(left_sub_tree_color);
        else
            return f2.get_result<int>(right_sub_tree_color);
    } else {
        // fprintf(stderr,/ "I am in get_coef task7\n");
        return -1;
    }

}

void diff_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {
    DiffArguments args = task->is_index_space ? *(const DiffArguments *) task->local_args
    : *(const DiffArguments *) task->args;

    int n = args.n;
    int l = args.l;
    int max_depth = args.max_depth;
    int actual_max_depth = args.actual_max_depth;
    int s0 = args.s0;
    int is_s0_valid = args.is_s0_valid;
    int RANDOM = 100;
    int sm, sp, r;

    DomainPoint my_sub_tree_color(Point<1>(0LL));
    DomainPoint left_sub_tree_color(Point<1>(1LL));
    DomainPoint right_sub_tree_color(Point<1>(2LL));
    Color partition_color1 = args.partition_color1;
    Color partition_color2 = args.partition_color2;

    // fprintf(stderr, "partition_color1 %d, partition_color2 %d\n", partition_color1, partition_color2);

    coord_t idx = args.idx;

    assert(regions.size() == 3);
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalRegion lr2 = regions[1].get_logical_region();
    LogicalRegion lr_whole = regions[2].get_logical_region();
    LogicalPartition lp = LogicalPartition::NO_PART, lp2 = LogicalPartition::NO_PART, lp11, lp21;

    fprintf(stderr, "step 1\n");

    lp = runtime->get_logical_partition_by_color(ctx, lr, partition_color1);
    fprintf(stderr, "step 2\n");
    LogicalRegion my_sub_tree_lr = runtime->get_logical_subregion_by_color(ctx, lp, my_sub_tree_color);
    LogicalRegion left_sub_tree_lr = runtime->get_logical_subregion_by_color(ctx, lp, left_sub_tree_color);
    LogicalRegion right_sub_tree_lr = runtime->get_logical_subregion_by_color(ctx, lp, right_sub_tree_color);


    LogicalRegion my_sub_tree_lr2 = LogicalRegion::NO_REGION;
    LogicalRegion left_sub_tree_lr2 = LogicalRegion::NO_REGION;
    LogicalRegion right_sub_tree_lr2 = LogicalRegion::NO_REGION;

    IndexSpace indexspace_left = left_sub_tree_lr.get_index_space();
    IndexSpace indexspace_right = right_sub_tree_lr.get_index_space();

    assert(lp != LogicalPartition::NO_PART);
    Rect<1> launch_domain(left_sub_tree_color, right_sub_tree_color);
    ArgumentMap arg_map;

    coord_t idx_left_sub_tree = idx + 1;
    coord_t idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));

    if (n < actual_max_depth) {
        IndexSpace is = lr2.get_index_space();
        DomainPointColoring coloring;

        Rect<1> my_sub_tree_rect(idx, idx);
        Rect<1> left_sub_tree_rect(idx_left_sub_tree, idx_right_sub_tree - 1);
        Rect<1> right_sub_tree_rect(idx_right_sub_tree,
        idx_right_sub_tree + static_cast<coord_t>(pow(2, max_depth - n)) - 2);

        coloring[my_sub_tree_color] = my_sub_tree_rect;
        coloring[left_sub_tree_color] = left_sub_tree_rect;
        coloring[right_sub_tree_color] = right_sub_tree_rect;

        Rect<1> color_space = Rect<1>(my_sub_tree_color, right_sub_tree_color);

        IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, coloring, DISJOINT_KIND, partition_color2);
        lp2 = runtime->get_logical_partition(ctx, lr2, ip);
        my_sub_tree_lr2 = runtime->get_logical_subregion_by_color(ctx, lp2, my_sub_tree_color);
        left_sub_tree_lr2 = runtime->get_logical_subregion_by_color(ctx, lp2, left_sub_tree_color);
        right_sub_tree_lr2 = runtime->get_logical_subregion_by_color(ctx, lp2, right_sub_tree_color);
    } else {
        return;
    }



    assert(my_sub_tree_lr2 != LogicalRegion::NO_REGION);
    assert(left_sub_tree_lr2 != LogicalRegion::NO_REGION);
    assert(right_sub_tree_lr2 != LogicalRegion::NO_REGION);
    assert(my_sub_tree_lr != LogicalRegion::NO_REGION);
    assert(left_sub_tree_lr != LogicalRegion::NO_REGION);
    assert(right_sub_tree_lr != LogicalRegion::NO_REGION);
    assert(lp2 != LogicalPartition::NO_PART);

    // fprintf(stderr, "step 1\n");

    if (is_s0_valid == false) {

        // fprintf(stderr, "step 2\n");

        if (runtime->has_index_partition(ctx, indexspace_left, partition_color1)) {

            // fprintf(stderr, "step 3\n");
            {
                // fprintf(stderr, "step 5\n");
                DiffSetTaskArgs args(idx, 0);
                TaskLauncher diff_set_task_launcher(DIFF_SET_TASK_ID, TaskArgument(&args, sizeof(DiffSetTaskArgs)));
                RegionRequirement req(my_sub_tree_lr2, WRITE_DISCARD, EXCLUSIVE, lr2);
                req.add_field(FID_X);
                diff_set_task_launcher.add_region_requirement(req);
                runtime->execute_task(ctx, diff_set_task_launcher);
                // fprintf(stderr, "step 13\n");
            }
            // fprintf(stderr, "step 7\n");

            DiffArguments for_left_sub_tree (n + 1, l * 2, max_depth, idx_left_sub_tree, partition_color1, partition_color2, actual_max_depth, RANDOM, false);

            // fprintf(stderr, "step 8\n");

            assert(my_sub_tree_lr2 != LogicalRegion::NO_REGION);
            assert(lr2 != LogicalRegion::NO_REGION);
            assert(lr != LogicalRegion::NO_REGION);
            assert(lr_whole != LogicalRegion::NO_REGION);
            assert(lp != LogicalPartition::NO_PART);
            assert(lp2 != LogicalPartition::NO_PART);


            TaskLauncher diff_launcher(DIFF_TASK_ID, TaskArgument(&for_left_sub_tree, sizeof(DiffArguments)));
            RegionRequirement req(left_sub_tree_lr, READ_ONLY, EXCLUSIVE, lr);
            RegionRequirement req2(left_sub_tree_lr2, WRITE_DISCARD, EXCLUSIVE, lr2);
            RegionRequirement req3(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole);
            req.add_field(FID_X);
            req2.add_field(FID_X);
            req3.add_field(FID_X);
            diff_launcher.add_region_requirement(req);
            diff_launcher.add_region_requirement(req2);
            diff_launcher.add_region_requirement(req3);
            runtime->execute_task(ctx, diff_launcher);
            // fprintf(stderr, "step 9\n");
        }
        if (runtime->has_index_partition(ctx, indexspace_right, partition_color1)) {
            // fprintf(stderr, "step 4\n");
            {
                DiffSetTaskArgs args(idx, 0);
                TaskLauncher diff_set_task_launcher(DIFF_SET_TASK_ID, TaskArgument(&args, sizeof(DiffSetTaskArgs)));
                RegionRequirement req(my_sub_tree_lr2, WRITE_DISCARD, EXCLUSIVE, lr2);
                req.add_field(FID_X);
                diff_set_task_launcher.add_region_requirement(req);
                runtime->execute_task(ctx, diff_set_task_launcher);
            }

            DiffArguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, partition_color1, partition_color2, actual_max_depth, RANDOM, false);

            TaskLauncher diff_launcher(DIFF_TASK_ID, TaskArgument(&for_right_sub_tree, sizeof(DiffArguments)));
            RegionRequirement req(right_sub_tree_lr, READ_ONLY, EXCLUSIVE, lr);
            RegionRequirement req2(right_sub_tree_lr2, WRITE_DISCARD, EXCLUSIVE, lr2);
            RegionRequirement req3(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole);
            req.add_field(FID_X);
            req2.add_field(FID_X);
            req3.add_field(FID_X);
            diff_launcher.add_region_requirement(req);
            diff_launcher.add_region_requirement(req2);
            diff_launcher.add_region_requirement(req3);
            runtime->execute_task(ctx, diff_launcher);
        } 
        if (runtime->has_index_partition(ctx, indexspace_left, partition_color1) == false &&
            runtime->has_index_partition(ctx, indexspace_right, partition_color1) == false) {
            Future f_s0;
            {
                ReadTaskArgs args(idx);
                TaskLauncher read_task_launcher(READ_TASK_ID, TaskArgument(&args, sizeof(ReadTaskArgs)));
                RegionRequirement req(my_sub_tree_lr, READ_ONLY, EXCLUSIVE, lr);
                req.add_field(FID_X);
                read_task_launcher.add_region_requirement(req);
                f_s0 = runtime->execute_task(ctx, read_task_launcher);
            }

            s0 = f_s0.get_result<int>();

            fprintf(stderr, "n is %d , l is %d\n", n, l);

            GetCoefArguments get_coef_args_sm(0, 0, max_depth, 0, partition_color1, n, l - 1);
            Future f_sm;
            {
                TaskLauncher get_coefs_launcher(GET_COEF_TASK_ID, TaskArgument(&get_coef_args_sm, sizeof(GetCoefArguments)));
                get_coefs_launcher.add_region_requirement(RegionRequirement(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole));
                get_coefs_launcher.add_field(0, FID_X);
                f_sm = runtime->execute_task(ctx, get_coefs_launcher);
            }
            sm = f_sm.get_result<int>();

            GetCoefArguments get_coef_args_sp(0, 0, max_depth, 0, partition_color1, n, l + 1);
            Future f_sp;
            {
                TaskLauncher get_coefs_launcher(GET_COEF_TASK_ID, TaskArgument(&get_coef_args_sp, sizeof(GetCoefArguments)));
                get_coefs_launcher.add_region_requirement(RegionRequirement(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole));
                get_coefs_launcher.add_field(0, FID_X);
                f_sp = runtime->execute_task(ctx, get_coefs_launcher);
            }
            sp = f_sp.get_result<int>();

            r = 0;

            if (sm >= 0 && sp >= 0 && s0 >= 0) {
                r = sm + sp + s0;
            }
            {
                DiffSetTaskArgs args(idx, r);
                TaskLauncher diff_set_task_launcher(DIFF_SET_TASK_ID, TaskArgument(&args, sizeof(DiffSetTaskArgs)));
                RegionRequirement req(my_sub_tree_lr2, WRITE_DISCARD, EXCLUSIVE, lr2);
                req.add_field(FID_X);
                diff_set_task_launcher.add_region_requirement(req);
                runtime->execute_task(ctx, diff_set_task_launcher);
            }

            if (sm < 0 || sp < 0 || s0 < 0) {
                DiffArguments for_left_sub_tree (n + 1, l * 2, max_depth, idx_left_sub_tree, partition_color1, partition_color2, actual_max_depth, s0/2, true);
                DiffArguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, partition_color1, partition_color2, actual_max_depth, s0/2, true);

                TaskLauncher diff_launcher_left(DIFF_TASK_ID, TaskArgument(&for_left_sub_tree, sizeof(DiffArguments)));
                RegionRequirement req_left(left_sub_tree_lr, READ_ONLY, EXCLUSIVE, lr);
                RegionRequirement req_left2(left_sub_tree_lr2, WRITE_DISCARD, EXCLUSIVE, lr2);
                RegionRequirement req_left3(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole);
                req_left.add_field(FID_X);
                req_left2.add_field(FID_X);
                req_left3.add_field(FID_X);
                diff_launcher_left.add_region_requirement(req_left);
                diff_launcher_left.add_region_requirement(req_left2);
                diff_launcher_left.add_region_requirement(req_left3);
                runtime->execute_task(ctx, diff_launcher_left);

                // TaskLauncher diff_launcher_right(DIFF_TASK_ID, TaskArgument(&for_right_sub_tree, sizeof(DiffArguments)));
                // RegionRequirement req_right(right_sub_tree_lr, READ_ONLY, EXCLUSIVE, lr);
                // RegionRequirement req_right2(right_sub_tree_lr2, WRITE_DISCARD, EXCLUSIVE, lr2);
                // RegionRequirement req_right3(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole);
                // req_right.add_field(FID_X);
                // req_right2.add_field(FID_X);
                // req_right3.add_field(FID_X);
                // diff_launcher_right.add_region_requirement(req_right);
                // diff_launcher_right.add_region_requirement(req_right2);
                // diff_launcher_right.add_region_requirement(req_right3);
                // runtime->execute_task(ctx, diff_launcher_right);


                // arg_map.set_point(left_sub_tree_color, TaskArgument(&for_left_sub_tree, sizeof(DiffArguments)));
                // arg_map.set_point(right_sub_tree_color, TaskArgument(&for_right_sub_tree, sizeof(DiffArguments)));

                // IndexTaskLauncher diff_launcher(DIFF_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
                // RegionRequirement req(lp, 0, READ_ONLY, EXCLUSIVE, lr);
                // RegionRequirement req2(lp2, 0, WRITE_DISCARD, EXCLUSIVE, lr2);
                // RegionRequirement req3(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole);
                // req.add_field(FID_X);
                // req2.add_field(FID_X);
                // req3.add_field(FID_X);
                // diff_launcher.add_region_requirement(req);
                // diff_launcher.add_region_requirement(req2);
                // diff_launcher.add_region_requirement(req3);
                // runtime->execute_index_space(ctx, diff_launcher);
            }
        }
        // fprintf(stderr, "step 10\n");
    // } else {
    //     // fprintf(stderr, "step 12\n");
    //     if (l % 2 == 0) {
    //         sp = s0;
    //         GetCoefArguments get_coef_args_sm(0, 0, max_depth, 0, partition_color1, n, l - 1);
    //         Future f_sm;
    //         {
    //             TaskLauncher get_coefs_launcher(GET_COEF_TASK_ID, TaskArgument(&get_coef_args_sm, sizeof(GetCoefArguments)));
    //             get_coefs_launcher.add_region_requirement(RegionRequirement(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole));
    //             get_coefs_launcher.add_field(0, FID_X);
    //             f_sm = runtime->execute_task(ctx, get_coefs_launcher);
    //         }
    //         sm = f_sm.get_result<int>();
    //     } else {
    //         sm = s0;
    //         GetCoefArguments get_coef_args_sp(0, 0, max_depth, 0, partition_color1, n, l + 1);
    //         Future f_sp;
    //         {
    //             TaskLauncher get_coefs_launcher(GET_COEF_TASK_ID, TaskArgument(&get_coef_args_sp, sizeof(GetCoefArguments)));
    //             get_coefs_launcher.add_region_requirement(RegionRequirement(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole));
    //             get_coefs_launcher.add_field(0, FID_X);
    //             f_sp = runtime->execute_task(ctx, get_coefs_launcher);
    //         }
    //         sp = f_sp.get_result<int>();
    //     }

    //     r = 0;

    //     if (sm >= 0 && sp >= 0 && s0 >= 0) {
    //         r = sm + sp + s0;
    //     }
    //     {
    //         DiffSetTaskArgs args(idx, r);
    //         TaskLauncher diff_set_task_launcher(DIFF_SET_TASK_ID, TaskArgument(&args, sizeof(DiffSetTaskArgs)));
    //         RegionRequirement req(my_sub_tree_lr2, WRITE_DISCARD, EXCLUSIVE, lr2);
    //         req.add_field(FID_X);
    //         diff_set_task_launcher.add_region_requirement(req);
    //         runtime->execute_task(ctx, diff_set_task_launcher);
    //     }


    //     if (sm < 0 || sp < 0 || s0 < 0) {
    //         DiffArguments for_left_sub_tree (n + 1, l * 2    , max_depth, idx_left_sub_tree, partition_color1, partition_color2, actual_max_depth, s0/2, true);
    //         DiffArguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, partition_color1, partition_color2, actual_max_depth, s0/2, true);

    //         arg_map.set_point(left_sub_tree_color, TaskArgument(&for_left_sub_tree, sizeof(DiffArguments)));
    //         arg_map.set_point(right_sub_tree_color, TaskArgument(&for_right_sub_tree, sizeof(DiffArguments)));



    //         DiffArguments for_left_sub_tree (n + 1, l * 2    , max_depth, idx_left_sub_tree, partition_color1, partition_color2, actual_max_depth, s0/2, true);
    //         DiffArguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, partition_color1, partition_color2, actual_max_depth, s0/2, true);

    //         TaskLauncher diff_launcher_left(DIFF_TASK_ID, TaskArgument(&for_left_sub_tree, sizeof(DiffArguments)));
    //         RegionRequirement req_left(left_sub_tree_lr, READ_ONLY, EXCLUSIVE, lr);
    //         RegionRequirement req_left2(left_sub_tree_lr2, WRITE_DISCARD, EXCLUSIVE, lr2);
    //         RegionRequirement req_left3(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole);
    //         req_left.add_field(FID_X);
    //         req_left2.add_field(FID_X);
    //         req_left3.add_field(FID_X);
    //         diff_launcher_left.add_region_requirement(req_left);
    //         diff_launcher_left.add_region_requirement(req_left2);
    //         diff_launcher_left.add_region_requirement(req_left3);
    //         runtime->execute_task(ctx, diff_launcher_left);

    //         TaskLauncher diff_launcher_right(DIFF_TASK_ID, TaskArgument(&for_right_sub_tree, sizeof(DiffArguments)));
    //         RegionRequirement req_right(right_sub_tree_lr, READ_ONLY, EXCLUSIVE, lr);
    //         RegionRequirement req_right2(right_sub_tree_lr2, WRITE_DISCARD, EXCLUSIVE, lr2);
    //         RegionRequirement req_right3(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole);
    //         req_right.add_field(FID_X);
    //         req_right2.add_field(FID_X);
    //         req_right3.add_field(FID_X);
    //         diff_launcher_right.add_region_requirement(req_right);
    //         diff_launcher_right.add_region_requirement(req_right2);
    //         diff_launcher_right.add_region_requirement(req_right3);
    //         runtime->execute_task(ctx, diff_launcher_right);


    //         // IndexTaskLauncher diff_launcher(DIFF_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
    //         // RegionRequirement req(lp, 0, READ_ONLY, EXCLUSIVE, lr);
    //         // RegionRequirement req2(lp2, 0, WRITE_DISCARD, EXCLUSIVE, lr2);
    //         // RegionRequirement req3(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole);
    //         // req.add_field(FID_X);
    //         // req2.add_field(FID_X);
    //         // req3.add_field(FID_X);
    //         // diff_launcher.add_region_requirement(req);
    //         // diff_launcher.add_region_requirement(req2);
    //         // diff_launcher.add_region_requirement(req3);
    //         // runtime->execute_index_space(ctx, diff_launcher);
    //     }

    }
    // fprintf(stderr, "step 11\n");
}

void print_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctxt, HighLevelRuntime *runtime) {

    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;

    int n = args.n,
    l = args.l,
    max_depth = args.max_depth;

    DomainPoint my_sub_tree_color(Point<1>(0LL));
    DomainPoint left_sub_tree_color(Point<1>(1LL));
    DomainPoint right_sub_tree_color(Point<1>(2LL));
    Color partition_color = args.partition_color;

    coord_t idx = args.idx;

    LogicalRegion lr = regions[0].get_logical_region();

    LogicalPartition lp = LogicalPartition::NO_PART;

    lp = runtime->get_logical_partition_by_color(ctxt, lr, partition_color);

    LogicalRegion my_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, my_sub_tree_color);
    LogicalRegion left_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, left_sub_tree_color);

    Future f1;
    {
        ReadTaskArgs args(idx);
        TaskLauncher read_task_launcher(READ_TASK_ID, TaskArgument(&args, sizeof(ReadTaskArgs)));
        RegionRequirement req(my_sub_tree_lr, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(FID_X);
        read_task_launcher.add_region_requirement(req);
        f1 = runtime->execute_task(ctxt, read_task_launcher);
    }

    int node_value = f1.get_result<int>();

    fprintf(stderr, "(n: %d, l: %d), idx: %lld, node_value: %d\n", n, l, idx, node_value);

    IndexSpace indexspace_left = left_sub_tree_lr.get_index_space();



    // These lines will create an instance for the whole region even though we need only the first element
    // const FieldAccessor<READ_ONLY, int, 1> read_acc(regions[0], FID_X);
    // int node_value = read_acc[idx];

    // checking if the children of the node have any valid partition. This condition implies that we are checking if we have reached the leaf node or not
    if (runtime->has_index_partition(ctxt, indexspace_left, partition_color)) {

        coord_t idx_left_sub_tree = idx + 1;
        coord_t idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));

        Rect<1> launch_domain(left_sub_tree_color, right_sub_tree_color);
        ArgumentMap arg_map;

        Arguments for_left_sub_tree(n + 1, 2 * l, max_depth, idx_left_sub_tree, partition_color);
        Arguments for_right_sub_tree(n + 1, 2 * l + 1, max_depth, idx_right_sub_tree, partition_color);

        arg_map.set_point(left_sub_tree_color, TaskArgument(&for_left_sub_tree, sizeof(Arguments)));
        arg_map.set_point(right_sub_tree_color, TaskArgument(&for_right_sub_tree, sizeof(Arguments)));

        // It calls print task twice for both the sub lp's lp[1], lp[2]
        IndexTaskLauncher print_launcher(PRINT_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);

        // We should not create a new partition, instead just fetch the existing partition, to avoid creating copy of the whole tree again and again
        // this partition color is the same color that we specified in the refine task while creating the index partition
        RegionRequirement req(lp, 0, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(FID_X);
        print_launcher.add_region_requirement(req);

        runtime->execute_index_space(ctxt, print_launcher);
    }
   
}

int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }

    {
        TaskVariantRegistrar registrar(REFINE_TASK_ID, "refine");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_inner(true);
        Runtime::preregister_task_variant<refine_task>(registrar, "refine");
    }

    {
        TaskVariantRegistrar registrar(SET_TASK_ID, "set");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<set_task>(registrar, "set");
    }

    {
        TaskVariantRegistrar registrar(PRINT_TASK_ID, "print");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<print_task>(registrar, "print");
    }

    {
        TaskVariantRegistrar registrar(READ_TASK_ID, "read");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<int, read_task>(registrar, "read");
    }

    {
        TaskVariantRegistrar registrar(COMPRESS_TASK_ID, "compress");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_inner(true);
        Runtime::preregister_task_variant<compress_task>(registrar, "compress");
    }

    {
        TaskVariantRegistrar registrar(COMPRESS_SET_TASK_ID, "compress_set");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<compress_set_task>(registrar, "compress_set");
    }

    {
        TaskVariantRegistrar registrar(GET_COEF_TASK_ID, "get_coef");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_inner(true);
        Runtime::preregister_task_variant<int, get_coef_task>(registrar, "get_coef");
    }

    {
        TaskVariantRegistrar registrar(DIFF_TASK_ID, "diff");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_inner(true);
        Runtime::preregister_task_variant<diff_task>(registrar, "diff");
    }

    {
        TaskVariantRegistrar registrar(DIFF_SET_TASK_ID, "diff_set");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<diff_set_task>(registrar, "diff_set");
    }

    return Runtime::start(argc, argv);
}