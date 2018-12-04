#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath> // pow
#include "legion.h"
#include "legion/legion_stl.h"
#include <vector>
#include <set>
#include <queue>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

using namespace Legion;
using namespace Legion::STL;

enum TASK_IDs {
    TOP_LEVEL_TASK_ID,
    UPPER_REFINE_TASK_ID,
    SET_TASK_ID,
    PRINT_TASK_ID,
    READ_TASK_ID,
    COMPRESS_TASK_ID,
    COMPRESS_SET_TASK_ID,
    GET_COEF_TASK_ID,
    DIFF_TASK_ID,
    DIFF_SET_TASK_ID,
    GET_COEF_UTIL_TASK_ID,
    INNER_PRODUCT_TASK_ID,
    PRODUCT_TASK_ID,
    NORM_TASK_ID,
    GAXPY_TASK_ID,
    GAXPY_SET_TASK_ID,
    RECONSTRUCT_SET_TASK_ID,
    RECONSTRUCT_TASK_ID,
    OUTER_REFINE_TASK_ID,
    SUB_TASKS_REFINE_TASK_ID,
    DUMMY_COMPRESS_TASK_ID,
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

    int tiling_height;

    Arguments(int _n, int _l, int _max_depth, coord_t _idx, Color _partition_color, int _actual_max_depth=0, int _tiling_height=0)
        : n(_n), l(_l), max_depth(_max_depth), idx(_idx), partition_color(_partition_color), actual_max_depth(_actual_max_depth),
          tiling_height(_tiling_height)
    {
        if (_actual_max_depth == 0) {
            actual_max_depth = _max_depth;
        }
    }
};

struct GaxpyArguments {
    int n;
    int l;
    int max_depth;
    coord_t idx;
    drand48_data gen;
    Color partition_color1, partition_color2, partition_color3;
    int actual_max_depth, left_tree_depth, right_tree_depth;

    GaxpyArguments(int _n, int _l, int _max_depth, coord_t _idx, Color _partition_color1, Color _partition_color2, Color _partition_color3, int _actual_max_depth, int _left_tree_depth, int _right_tree_depth)
        : n(_n), l(_l), max_depth(_max_depth), idx(_idx), partition_color1(_partition_color1),
        partition_color2(_partition_color2), partition_color3(_partition_color3),
        actual_max_depth(_actual_max_depth), left_tree_depth(_left_tree_depth), 
        right_tree_depth(_right_tree_depth)
    {}
};

struct SetTaskArgs {
    int node_value;
    coord_t idx;
    int n;
    int max_depth;
    SetTaskArgs(int _node_value, coord_t _idx, int _n, int _max_depth) : node_value(_node_value), idx(_idx), n(_n), max_depth(_max_depth) {}
};

struct GaxpySetTaskArgs {
    coord_t idx;
    bool is_left, is_right;
    GaxpySetTaskArgs(coord_t _idx, bool _is_left, bool _is_right) : idx(_idx), is_left(_is_left), is_right(_is_right) {}
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
    int n, l, max_depth;
    coord_t idx;
    Color partition_color1;
    Color partition_color2;
    int actual_max_depth;
    int s0;
    bool is_s0_valid;
    
    DiffArguments(int _n, int _l, int _max_depth, coord_t _idx, Color _partition_color1, Color _partition_color2, int _actual_max_depth, int _s0, bool _is_s0_valid)
        : n(_n), l(_l), max_depth(_max_depth), idx(_idx), partition_color1(_partition_color1), partition_color2(_partition_color2),
        actual_max_depth(_actual_max_depth), s0(_s0), is_s0_valid(_is_s0_valid)
    {}
};

struct DiffSetTaskArgs {
    coord_t idx;
    int node_value;
    DiffSetTaskArgs(coord_t _idx, int _node_value) : 
        idx(_idx), node_value(_node_value) {}
};

struct GetCoefArguments {
    int n, l, max_depth;
    coord_t idx;
    Color partition_color;
    int questioned_n, questioned_l;
    
    GetCoefArguments(int _n, int _l, int _max_depth, coord_t _idx, Color _partition_color, int _questioned_n, int _questioned_l)
        : n(_n), l(_l), max_depth(_max_depth), idx(_idx), partition_color(_partition_color),
        questioned_n(_questioned_n), questioned_l(_questioned_l)
    {}
};

struct ReturnGetCoefArguments {
    int n, l;
    LogicalRegion lr;
    coord_t idx;
    bool exists;
    ReturnGetCoefArguments(int _n, int _l, LogicalRegion _lr, coord_t _idx, bool _exists)
        : n(_n), l(_l), lr(_lr), idx(_idx), exists(_exists)

    {}
};

struct GetCoefUtilArguments {
    int n, l, max_depth;
    coord_t idx;
    Color partition_color;
    int questioned_n, questioned_l;

    struct ReturnGetCoefArguments parent;

    GetCoefUtilArguments(int _n, int _l, int _max_depth, coord_t _idx, Color _partition_color, int _questioned_n, int _questioned_l, struct ReturnGetCoefArguments _parent)
        : n(_n), l(_l), max_depth(_max_depth), idx(_idx), partition_color(_partition_color),
        questioned_n(_questioned_n), questioned_l(_questioned_l), parent(_parent)

    {}
};

struct InnerProductArguments {
    int n;
    int l;
    int max_depth;
    coord_t idx;
    drand48_data gen;
    Color partition_color1, partition_color2;
    int actual_max_depth;

    InnerProductArguments(int _n, int _l, int _max_depth, coord_t _idx, Color _partition_color1, Color _partition_color2, int _actual_max_depth)
        : n(_n), l(_l), max_depth(_max_depth), idx(_idx), partition_color1(_partition_color1),
        partition_color2(_partition_color2), actual_max_depth(_actual_max_depth)
    {}
};

struct ReConstructArguments {
    int n, l, max_depth;
    coord_t idx;
    drand48_data gen;
    Color partition_color;
    int parent_value;
    ReConstructArguments(int _n, int _l, int _max_depth, coord_t _idx, Color _partition_color, int _parent_value)
        : n(_n), l(_l), max_depth(_max_depth), idx(_idx), partition_color(_partition_color),
        parent_value(_parent_value)
    {}
};

struct ReConstructSetTaskArgs {
    coord_t idx;
    int node_value;
    ReConstructSetTaskArgs(coord_t _idx, int _node_value) : 
        idx(_idx), node_value(_node_value){}
};

struct ReturnRefineArguments {
    int n, l;
    coord_t idx;
    int node_value;
    ReturnRefineArguments(int _n, int _l, coord_t _idx, int _node_value) : 
        n(_n), l(_l), idx(_idx), node_value(_node_value){}
};

void reconstruct_set_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, HighLevelRuntime *runtime) {

    ReConstructSetTaskArgs args = *(const ReConstructSetTaskArgs *) task->args;
    assert(regions.size() == 1);
    const FieldAccessor<READ_WRITE, int, 1> write_acc(regions[0], FID_X);

    write_acc[args.idx] = args.node_value;
}

struct ReturnRefineTaskArgs {
    int n, l;
    coord_t idx;
    int sub_tree_num;
    ReturnRefineTaskArgs(int _n, int _l, coord_t _idx, int _sub_tree_num=0)
        : n(_n), l(_l), idx(_idx), sub_tree_num(_sub_tree_num) {}

    ReturnRefineTaskArgs() {}
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

    int overall_max_depth = 5;
    int actual_left_depth = 5;

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

    // For 1st logical region
    LogicalRegion lr1 = runtime->create_logical_region(ctx, is, fs);
    // Any random value will work
    Color partition_color1 = 10;
    int tiling_height = 3;

    Arguments args1(0, 0, overall_max_depth, 0, partition_color1, actual_left_depth, tiling_height);
    srand48_r(seed, &args1.gen);

    // Launching the refine task
    TaskLauncher main_refine_launcher(OUTER_REFINE_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    main_refine_launcher.add_region_requirement(RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1));
    main_refine_launcher.add_field(0, FID_X);
    runtime->execute_task(ctx, main_refine_launcher);



    // // Launching another task to print the values of the binary tree nodes
    // TaskLauncher print_launcher(PRINT_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    // print_launcher.add_region_requirement(RegionRequirement(lr1, READ_ONLY, EXCLUSIVE, lr1));
    // print_launcher.add_field(0, FID_X);
    // runtime->execute_task(ctx, print_launcher);

    // Launching another task to print the values of the binary tree nodes
    args1.idx = 3;
    TaskLauncher dummy_compress_launcher(DUMMY_COMPRESS_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    dummy_compress_launcher.add_region_requirement(RegionRequirement(lr1, READ_WRITE, EXCLUSIVE, lr1));
    dummy_compress_launcher.add_field(0, FID_X);
    runtime->execute_task(ctx, dummy_compress_launcher);

    // // Launching another task to print the values of the binary tree nodes
    // TaskLauncher print_launcher1(PRINT_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    // print_launcher1.add_region_requirement(RegionRequirement(lr1, READ_ONLY, EXCLUSIVE, lr1));
    // print_launcher1.add_field(0, FID_X);
    // runtime->execute_task(ctx, print_launcher1);

    // TaskLauncher norm_launcher(NORM_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    // norm_launcher.add_region_requirement(RegionRequirement(lr1, READ_ONLY, EXCLUSIVE, lr1));
    // norm_launcher.add_field(0, FID_X);
    // Future f1 = runtime->execute_task(ctx, norm_launcher);
    // float norm_value = sqrt(f1.get_result<int>());
    // fprintf(stderr, "norm result %fm\n", norm_value);

    // For 2nd logical region
    // int actual_right_depth = 6;
    // Rect<1> tree_rect2(0LL, static_cast<coord_t>(pow(2, overall_max_depth + 1)) - 2);
    // IndexSpace is2 = runtime->create_index_space(ctx, tree_rect2);

    // LogicalRegion lr2 = runtime->create_logical_region(ctx, is2, fs);
    // // Any random value will work
    // Color partition_color2 = 20;

    // Arguments args2(0, 0, overall_max_depth, 0, partition_color2, actual_right_depth);
    // srand48_r(seed, &args2.gen);

    // // Launching the refine task
    // TaskLauncher refine_launcher2(REFINE_TASK_ID, TaskArgument(&args2, sizeof(Arguments)));
    // refine_launcher2.add_region_requirement(RegionRequirement(lr2, WRITE_DISCARD, EXCLUSIVE, lr2));
    // refine_launcher2.add_field(0, FID_X);
    // runtime->execute_task(ctx, refine_launcher2);

    // // Launching another task to print the values of the binary tree nodes
    // TaskLauncher print_launcher2_1(PRINT_TASK_ID, TaskArgument(&args2, sizeof(Arguments)));
    // print_launcher2_1.add_region_requirement(RegionRequirement(lr2, READ_ONLY, EXCLUSIVE, lr2));
    // print_launcher2_1.add_field(0, FID_X);
    // runtime->execute_task(ctx, print_launcher2_1);

    // Launching another task to print the values of the binary tree nodes
    // TaskLauncher compress_launcher2(COMPRESS_TASK_ID, TaskArgument(&args2, sizeof(Arguments)));
    // compress_launcher2.add_region_requirement(RegionRequirement(lr2, READ_WRITE, EXCLUSIVE, lr2));
    // compress_launcher2.add_field(0, FID_X);
    // runtime->execute_task(ctx, compress_launcher2);

    // Launching another task to print the values of the binary tree nodes
    // TaskLauncher print_launcher2_2(PRINT_TASK_ID, TaskArgument(&args2, sizeof(Arguments)));
    // print_launcher2_2.add_region_requirement(RegionRequirement(lr2, READ_ONLY, EXCLUSIVE, lr2));
    // print_launcher2_2.add_field(0, FID_X);
    // runtime->execute_task(ctx, print_launcher2_2);

    // Rect<1> dummy_tree_rect(0LL, static_cast<coord_t>(pow(2, overall_max_depth + 1)) - 2);
    // IndexSpace dummy_is = runtime->create_index_space(ctx, dummy_tree_rect);
    // LogicalRegion dummy_lr = runtime->create_logical_region(ctx, dummy_is, fs);

    // DiffArguments diff_args(0, 0, overall_max_depth, 0, partition_color1, partition_color2, actual_left_depth, 100, false);

    // TaskLauncher diff_launcher(DIFF_TASK_ID, TaskArgument(&diff_args, sizeof(DiffArguments)));
    // diff_launcher.add_region_requirement(RegionRequirement(lr1, READ_ONLY, EXCLUSIVE, lr1));
    // diff_launcher.add_region_requirement(RegionRequirement(lr2, WRITE_DISCARD, EXCLUSIVE, lr2));
    // diff_launcher.add_region_requirement(RegionRequirement(lr1, READ_ONLY, EXCLUSIVE, lr1));
    // diff_launcher.add_region_requirement(RegionRequirement(dummy_lr, READ_ONLY, EXCLUSIVE, dummy_lr));
    // diff_launcher.add_field(0, FID_X);
    // diff_launcher.add_field(1, FID_X);
    // diff_launcher.add_field(2, FID_X);
    // diff_launcher.add_field(3, FID_X);
    // runtime->execute_task(ctx, diff_launcher);

    // Arguments args2(0, 0, overall_max_depth, 0, partition_color2, actual_left_depth);

    // // Launching another task to print the values of the binary tree nodes
    // TaskLauncher print_launcher12(PRINT_TASK_ID, TaskArgument(&args2, sizeof(Arguments)));
    // print_launcher12.add_region_requirement(RegionRequirement(lr2, READ_ONLY, EXCLUSIVE, lr2));
    // print_launcher12.add_field(0, FID_X);
    // runtime->execute_task(ctx, print_launcher12);

    // InnerProductArguments args3_inner_product(0, 0, overall_max_depth, 0, partition_color1, partition_color2, min(actual_left_depth, actual_right_depth));

    // // Launching inner product task
    // TaskLauncher inner_product_launcher(INNER_PRODUCT_TASK_ID, TaskArgument(&args3_inner_product, sizeof(InnerProductArguments)));
    // inner_product_launcher.add_region_requirement(RegionRequirement(lr1, READ_ONLY, EXCLUSIVE, lr1));
    // inner_product_launcher.add_region_requirement(RegionRequirement(lr2, READ_ONLY, EXCLUSIVE, lr2));
    // inner_product_launcher.add_field(0, FID_X);
    // inner_product_launcher.add_field(1, FID_X);

    // // inner_product_launcher.add_future(f1);
    // // inner_product_launcher.add_future(f2);

    // Future f_result = runtime->execute_task(ctx, inner_product_launcher);

    // fprintf(stderr, "inner product result %d\n", f_result.get_result<int>());

    // For 3rd logical region
    // int actual_new_tree_depth = max(actual_left_depth, actual_right_depth);

    // Rect<1> tree_rect3(0LL, static_cast<coord_t>(pow(2, overall_max_depth + 1)) - 2);
    // IndexSpace is3 = runtime->create_index_space(ctx, tree_rect3);
    // LogicalRegion lr3 = runtime->create_logical_region(ctx, is3, fs);

    // Color partition_color3 = 30;

    // Rect<1> dummy_tree_rect_gaxpy(0LL, static_cast<coord_t>(pow(2, overall_max_depth + 1)) - 2);
    // IndexSpace dummy_is_gaxpy = runtime->create_index_space(ctx, dummy_tree_rect_gaxpy);
    // LogicalRegion dummy_lr_gaxpy = runtime->create_logical_region(ctx, dummy_is_gaxpy, fs);

    // GaxpyArguments args3(0, 0, overall_max_depth, 0, partition_color1, partition_color2, partition_color3, actual_new_tree_depth, actual_left_depth, actual_right_depth);

    // // Launching gaxpy task 
    // TaskLauncher gaxpy_launcher(GAXPY_TASK_ID, TaskArgument(&args3, sizeof(GaxpyArguments)));
    // gaxpy_launcher.add_region_requirement(RegionRequirement(lr1, READ_ONLY, EXCLUSIVE, lr1));
    // gaxpy_launcher.add_region_requirement(RegionRequirement(lr2, READ_ONLY, EXCLUSIVE, lr2));
    // gaxpy_launcher.add_region_requirement(RegionRequirement(lr3, WRITE_DISCARD, EXCLUSIVE, lr3));
    // gaxpy_launcher.add_region_requirement(RegionRequirement(dummy_lr_gaxpy, READ_ONLY, EXCLUSIVE, dummy_lr_gaxpy));
    // gaxpy_launcher.add_field(0, FID_X);
    // gaxpy_launcher.add_field(1, FID_X);
    // gaxpy_launcher.add_field(2, FID_X);
    // gaxpy_launcher.add_field(3, FID_X);
    // runtime->execute_task(ctx, gaxpy_launcher);

    // Arguments args4(0, 0, overall_max_depth, 0, partition_color3, actual_new_tree_depth);

    // // Launching another task to print the values of the binary tree nodes
    // TaskLauncher print_launcher3_1(PRINT_TASK_ID, TaskArgument(&args4, sizeof(Arguments)));
    // print_launcher3_1.add_region_requirement(RegionRequirement(lr3, READ_ONLY, EXCLUSIVE, lr3));
    // print_launcher3_1.add_field(0, FID_X);
    // runtime->execute_task(ctx, print_launcher3_1);

    // ReConstructArguments reconstruct_args(0, 0, overall_max_depth, 0, partition_color1, 0);

    // // Launching another task to print the values of the binary tree nodes
    // TaskLauncher reconstruct_launcher(RECONSTRUCT_TASK_ID, TaskArgument(&reconstruct_args, sizeof(ReConstructArguments)));
    // reconstruct_launcher.add_region_requirement(RegionRequirement(lr1, READ_WRITE, EXCLUSIVE, lr1));
    // reconstruct_launcher.add_field(0, FID_X);
    // runtime->execute_task(ctx, reconstruct_launcher);

    // // Launching another task to print the values of the binary tree nodes
    // TaskLauncher print_launcher2(PRINT_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    // print_launcher2.add_region_requirement(RegionRequirement(lr1, READ_ONLY, EXCLUSIVE, lr1));
    // print_launcher2.add_field(0, FID_X);
    // runtime->execute_task(ctx, print_launcher2);

    // Destroying allocated memory
    // runtime->destroy_logical_region(ctx, lr1);
    // runtime->destroy_logical_region(ctx, lr2);
    // runtime->destroy_field_space(ctx, fs);
    // runtime->destroy_index_space(ctx, is);
    // runtime->destroy_index_space(ctx, is2);
}

void set_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {

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

void gaxpy_set_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {

    GaxpySetTaskArgs args = *(const GaxpySetTaskArgs *) task->args;
    assert(regions.size() == 3);

    const FieldAccessor<WRITE_DISCARD, int, 1> write_acc(regions[2], FID_X);
    write_acc[args.idx] = 0;

    if (args.is_right == true) {
        const FieldAccessor<READ_ONLY, int, 1> write_acc2(regions[1], FID_X);
        write_acc[args.idx] = write_acc[args.idx] + write_acc2[args.idx];
    }

    if (args.is_left == true) {
        const FieldAccessor<READ_ONLY, int, 1> write_acc1(regions[0], FID_X);
        write_acc[args.idx] = write_acc[args.idx] + write_acc1[args.idx];
    }

}

int read_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {

    ReadTaskArgs args = *(const ReadTaskArgs *) task->args;
    assert(regions.size() == 1);
    const FieldAccessor<READ_ONLY, int, 1> read_acc(regions[0], FID_X);
    return read_acc[args.idx];
}

void compress_set_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {
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
}


// To be recursive task calling for the left and right subtrees, if necessary !
vector<ReturnRefineTaskArgs> upper_refine_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {

    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    int n = args.n;
    int l = args.l;
    int max_depth = args.max_depth;
    int tiling_height = args.tiling_height;

    coord_t idx = args.idx;

    assert(regions.size() == 1);
    LogicalRegion lr = regions[0].get_logical_region();

    coord_t idx_left_sub_tree = 0LL;
    coord_t idx_right_sub_tree = 0LL;

    assert(lr != LogicalRegion::NO_REGION);

    long int node_value;
    lrand48_r(&args.gen, &node_value);
    node_value = node_value % 10 + 1;

    std::queue<ReturnRefineArguments> q;
    const FieldAccessor<WRITE_DISCARD, int, 1> write_acc(regions[0], FID_X);
    struct ReturnRefineArguments root(n, l, idx, node_value);
    q.push(root);

    while(n <= tiling_height-1 && !q.empty()) {
        struct ReturnRefineArguments val = q.front();
        q.pop();
        val.node_value = val.node_value % 10 + 1;
        if(val.node_value > 3) {
            write_acc[val.idx] = val.node_value;
            idx_left_sub_tree = 2*val.idx + 1;
            idx_right_sub_tree = 2*val.idx + 2;
            lrand48_r(&args.gen, &node_value);
            struct ReturnRefineArguments left_index(val.n+1, 2*val.l, idx_left_sub_tree, node_value);
            lrand48_r(&args.gen, &node_value);
            struct ReturnRefineArguments right_index(val.n+1, (2*val.l)+1, idx_right_sub_tree, node_value);
            q.push(left_index);
            q.push(right_index);
            n = n+1; 
        }
    }

    vector<ReturnRefineTaskArgs> f_result_value;
    int sub_tree_num = 1;
    
    if(n > tiling_height-1) {
        while(!q.empty()) {
            struct ReturnRefineArguments val = q.front();
            q.pop();
            if(val.node_value > 3) {
                int new_val = static_cast<coord_t>(pow(2, max_depth-(val.n+2))) + 1;
                idx_left_sub_tree = val.idx + static_cast<coord_t>(pow(2, val.n)) - val.l + (2 * val.l * new_val);
                idx_right_sub_tree = idx_left_sub_tree + new_val;
                lrand48_r(&args.gen, &node_value);
                write_acc[idx_left_sub_tree] = node_value;
                lrand48_r(&args.gen, &node_value);
                write_acc[idx_right_sub_tree] = node_value;
                ReturnRefineTaskArgs new_result(n+1, 2*l , idx_left_sub_tree, sub_tree_num);
                ReturnRefineTaskArgs new_result1(n+1, 2*l + 1 , idx_right_sub_tree, sub_tree_num + 1);
                f_result_value.push_back(new_result);
                f_result_value.push_back(new_result1);
            }
            sub_tree_num += 2;
            
        }
        
    }

    return f_result_value;
}

// To be recursive task calling for the left and right subtrees, if necessary !
void sub_tasks_refine_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {

    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    int n = args.n;
    int l = args.l;
    int actual_max_depth = args.actual_max_depth;

    coord_t idx = args.idx;

    assert(regions.size() == 1);
    LogicalRegion lr = regions[0].get_logical_region();

    coord_t idx_left_sub_tree = 0LL;
    coord_t idx_right_sub_tree = 0LL;
    
    assert(lr != LogicalRegion::NO_REGION);

    long int node_value;
    lrand48_r(&args.gen, &node_value);
    node_value = node_value % 10 + 1;

    std::queue<ReturnRefineArguments> q;
    const FieldAccessor<WRITE_DISCARD, int, 1> write_acc(regions[0], FID_X);
    struct ReturnRefineArguments root(n, l, idx, node_value);
    q.push(root);


    while(n < actual_max_depth && !q.empty()) {
        struct ReturnRefineArguments val = q.front();
        q.pop();
        if(val.node_value > 3) {
            write_acc[val.idx] = node_value % 10 + 1;
            idx_left_sub_tree = 2*val.idx + 1 - val.idx;
            idx_right_sub_tree = 2*val.idx + 2 - val.idx;
            lrand48_r(&args.gen, &node_value);
            struct ReturnRefineArguments left_index(n+1, 2*l, idx_left_sub_tree, node_value);
            lrand48_r(&args.gen, &node_value);
            struct ReturnRefineArguments right_index(n+1, (2*l)+1, idx_right_sub_tree, node_value);
            q.push(left_index);
            q.push(right_index);
            n = n+1;
        }
    }

}

void outer_refine_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;

    int tiling_height = args.tiling_height;
    int max_depth = args.max_depth;
    int actual_max_depth = args.actual_max_depth;

    args.actual_max_depth = tiling_height;
    LogicalRegion lr = regions[0].get_logical_region();

    // Launching the refine task
    TaskLauncher refine_launcher(UPPER_REFINE_TASK_ID, TaskArgument(&args, sizeof(Arguments)));
    refine_launcher.add_region_requirement(RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
    refine_launcher.add_field(0, FID_X);
    Future f_result = runtime->execute_task(ctx, refine_launcher);

    vector<ReturnRefineTaskArgs> potential_indexes = f_result.get_result< vector<ReturnRefineTaskArgs> >();

    std::cout<<"\n indexes size "<<potential_indexes.size();

    for(int i=0; i<potential_indexes.size(); i++)
        std::cout<<"\n index "<<potential_indexes[i].idx<<" sub tree num "<<potential_indexes[i].sub_tree_num;

    ArgumentMap arg_map;

    int h = pow(2, tiling_height);
    coord_t next_index = static_cast<coord_t>(pow(2, tiling_height) - 2);
    coord_t copy_idx = 0;

    IndexSpace is = lr.get_index_space();
    DomainPointColoring coloring;

    for(int i = 0; i <= h; i++) {
        coloring[Point<1>(i)] = Rect<1> (copy_idx, next_index);
        copy_idx = next_index + 1;
        next_index = copy_idx + static_cast<coord_t>(pow(2, max_depth-tiling_height) - 2);
        
    }

    Rect<1> color_space = Rect<1>(Point<1>(0LL), Point<1>(h));

    std::cout<<"\n partition_color3 "<<args.partition_color;

    IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, coloring, DISJOINT_KIND, args.partition_color);
    LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);

    LogicalPartition lp1 = runtime->get_logical_partition_by_color(ctx, lr, args.partition_color);

    assert(lp == lp1);

    assert(lp1 != LogicalPartition::NO_PART);

    for(int i=0; i<potential_indexes.size(); i++) {
        Arguments passing_args (potential_indexes[i].n, potential_indexes[i].l , args.max_depth, potential_indexes[i].idx, args.partition_color, actual_max_depth, tiling_height);
        // Launching the refine task for the sub tasks
        TaskLauncher refine_launcher_sub_tasks(SUB_TASKS_REFINE_TASK_ID, TaskArgument(&passing_args, sizeof(Arguments)));
        DomainPoint sub_tree_color(Point<1>(potential_indexes[i].sub_tree_num));
        LogicalRegion sub_lr = runtime->get_logical_subregion_by_color(ctx, lp, sub_tree_color);
        refine_launcher_sub_tasks.add_region_requirement(RegionRequirement(sub_lr, READ_WRITE, EXCLUSIVE, lr));
        refine_launcher_sub_tasks.add_field(0, FID_X);
        runtime->execute_task(ctx, refine_launcher_sub_tasks);
        is = sub_lr.get_index_space();
        runtime->create_index_partition(ctx, is, color_space, coloring, DISJOINT_KIND, args.partition_color);
    }

}

void reconstruct_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctxt, HighLevelRuntime *runtime) {
    ReConstructArguments args = task->is_index_space ? *(const ReConstructArguments *) task->local_args
    : *(const ReConstructArguments *) task->args;

    int n = args.n;
    int l = args.l;
    int max_depth = args.max_depth;
    int parent_value = args.parent_value;

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

    IndexSpace indexspace_left = left_sub_tree_lr.get_index_space();

    Future f1;
    {
        ReadTaskArgs args(idx);
        TaskLauncher read_task_launcher(READ_TASK_ID, TaskArgument(&args, sizeof(ReadTaskArgs)));
        RegionRequirement req(my_sub_tree_lr, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(FID_X);
        read_task_launcher.add_region_requirement(req);
        f1 = runtime->execute_task(ctxt, read_task_launcher);
    }

    parent_value = (parent_value + f1.get_result<int>())/2;

    if (runtime->has_index_partition(ctxt, indexspace_left, partition_color)) {
        idx_left_sub_tree = idx + 1;
        idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));

        {
            ReConstructSetTaskArgs args(idx, 0);
            TaskLauncher reconstruct_set_task_launcher(RECONSTRUCT_SET_TASK_ID, TaskArgument(&args, sizeof(ReConstructSetTaskArgs)));
            RegionRequirement req(my_sub_tree_lr, READ_WRITE, EXCLUSIVE, lr);
            req.add_field(FID_X);
            reconstruct_set_task_launcher.add_region_requirement(req);
            runtime->execute_task(ctxt, reconstruct_set_task_launcher);
        }

        Rect<1> launch_domain(left_sub_tree_color, right_sub_tree_color);
        ArgumentMap arg_map;

        ReConstructArguments for_left_sub_tree(n + 1, 2 * l, max_depth, idx_left_sub_tree, partition_color, parent_value);
        ReConstructArguments for_right_sub_tree(n + 1, 2 * l + 1, max_depth, idx_right_sub_tree, partition_color, parent_value);

        arg_map.set_point(left_sub_tree_color, TaskArgument(&for_left_sub_tree, sizeof(ReConstructArguments)));
        arg_map.set_point(right_sub_tree_color, TaskArgument(&for_right_sub_tree, sizeof(ReConstructArguments)));

        IndexTaskLauncher reconstruct_launcher(RECONSTRUCT_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        RegionRequirement req(lp, 0, READ_WRITE, EXCLUSIVE, lr);
        req.add_field(FID_X);
        reconstruct_launcher.add_region_requirement(req);
        runtime->execute_index_space(ctxt, reconstruct_launcher);

    } else {
        {
            ReConstructSetTaskArgs args(idx, parent_value);
            TaskLauncher reconstruct_set_task_launcher(RECONSTRUCT_SET_TASK_ID, TaskArgument(&args, sizeof(ReConstructSetTaskArgs)));
            RegionRequirement req(my_sub_tree_lr, READ_WRITE, EXCLUSIVE, lr);
            req.add_field(FID_X);
            reconstruct_set_task_launcher.add_region_requirement(req);
            runtime->execute_task(ctxt, reconstruct_set_task_launcher);
        }
    }
}

void compress_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctxt, HighLevelRuntime *runtime) {
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;

    int n = args.n;
    int l = args.l;
    int max_depth = args.max_depth;
    // int tiling_height = args.tiling_height;

    std::cout<<"\n here ";

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
    // LogicalRegion my_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, my_sub_tree_color);
    // LogicalRegion left_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, left_sub_tree_color);
    // LogicalRegion right_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, right_sub_tree_color);

    // IndexSpace indexspace_left = left_sub_tree_lr.get_index_space();

    // if (n < max_depth) {

    //     idx_left_sub_tree = 2*idx + 1 - idx;
    //     idx_right_sub_tree = 2*idx + 2 - idx;

    //     Rect<1> launch_domain(left_sub_tree_color, right_sub_tree_color);
    //     ArgumentMap arg_map;

    //     Arguments for_left_sub_tree(n + 1, 2 * l, max_depth, idx_left_sub_tree, partition_color);
    //     Arguments for_right_sub_tree(n + 1, 2 * l + 1, max_depth, idx_right_sub_tree, partition_color);

    //     arg_map.set_point(left_sub_tree_color, TaskArgument(&for_left_sub_tree, sizeof(Arguments)));
    //     arg_map.set_point(right_sub_tree_color, TaskArgument(&for_right_sub_tree, sizeof(Arguments)));

    //     IndexTaskLauncher compress_launcher(COMPRESS_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
    //     RegionRequirement req(lp, 0, READ_WRITE, EXCLUSIVE, lr);
    //     req.add_field(FID_X);
    //     compress_launcher.add_region_requirement(req);
    //     runtime->execute_index_space(ctxt, compress_launcher);

    //     {
    //         CompressSetTaskArgs args(idx, idx_left_sub_tree, idx_right_sub_tree);
    //         TaskLauncher compress_set_task_launcher(COMPRESS_SET_TASK_ID, TaskArgument(&args, sizeof(CompressSetTaskArgs)));
    //         RegionRequirement req(my_sub_tree_lr, READ_WRITE, EXCLUSIVE, lr);
    //         RegionRequirement req_left(left_sub_tree_lr, READ_WRITE, EXCLUSIVE, lr);
    //         RegionRequirement req_right(right_sub_tree_lr, READ_WRITE, EXCLUSIVE, lr);
    //         req.add_field(FID_X);
    //         req_left.add_field(FID_X);
    //         req_right.add_field(FID_X);
    //         compress_set_task_launcher.add_region_requirement(req);
    //         compress_set_task_launcher.add_region_requirement(req_left);
    //         compress_set_task_launcher.add_region_requirement(req_right);
    //         runtime->execute_task(ctxt, compress_set_task_launcher);
    //     }
    // }
}

void dummy_compress_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctxt, HighLevelRuntime *runtime) {
    std::cout<<"\n step0";
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;

    std::cout<<"\n step0.0";

    int n = args.n;
    int l = args.l;
    int max_depth = args.max_depth;
    int tiling_height = args.tiling_height;
    int actual_max_depth = args.actual_max_depth;

    Color partition_color = args.partition_color;

    coord_t idx = args.idx;

    std::cout<<"\n step0.3"<<" partition_color2 "<<partition_color;

    assert(regions.size() == 1);
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalPartition lp = LogicalPartition::NO_PART, lp1,lp2;
    lp = runtime->get_logical_partition_by_color(ctxt, lr, partition_color);

    std::cout<<"\n step0.4";

    int idx_left_sub_tree = 0LL;
    int idx_right_sub_tree = 0LL;
    long long int  idx_val = (long long int)args.idx;

    DomainPoint first_sub_tree_color(Point<1>(1LL));
    DomainPoint second_sub_tree_color(Point<1>(2LL));
    DomainPoint third_sub_tree_color(Point<1>(3LL));
    DomainPoint fourth_sub_tree_color(Point<1>(4LL));
    DomainPoint fifth_sub_tree_color(Point<1>(5LL));
    DomainPoint sixth_sub_tree_color(Point<1>(6LL));
    DomainPoint seventh_sub_tree_color(Point<1>(7LL));
    DomainPoint eighth_sub_tree_color(Point<1>(8LL));

    LogicalRegion first_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, first_sub_tree_color);
    LogicalRegion second_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, second_sub_tree_color);
    LogicalRegion third_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, third_sub_tree_color);
    LogicalRegion fourth_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, fourth_sub_tree_color);
    LogicalRegion fifth_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, fifth_sub_tree_color);
    LogicalRegion sixth_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, sixth_sub_tree_color);
    LogicalRegion seventh_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, seventh_sub_tree_color);
    LogicalRegion eighth_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, eighth_sub_tree_color);
    assert(first_sub_tree_lr != LogicalRegion::NO_REGION);


    std::cout<<"\n n "<<n<<" l "<<l<<" idx "<<idx<<" tiling_height "<<tiling_height;

    // if (n < tiling_height) {
    //     idx_left_sub_tree = 3;
    //     idx_right_sub_tree = 10;

    //     // Rect<1> launch_domain(Point<1>(0LL), Point<1>(8LL));
    //     ArgumentMap arg_map;

    //     Arguments for_left_sub_tree(n + 1, 2 * l, max_depth, idx_left_sub_tree, partition_color, actual_max_depth, tiling_height);

    //     TaskLauncher dummy_compress_launcher(DUMMY_COMPRESS_TASK_ID, TaskArgument(&for_left_sub_tree, sizeof(Arguments)));
    //     RegionRequirement req(left_sub_tree_lr, READ_ONLY, EXCLUSIVE, lr);
    //     req.add_field(FID_X);
    //     dummy_compress_launcher.add_region_requirement(req);
    //     std::cout<<"\n step2";
    //     runtime->execute_task(ctxt, dummy_compress_launcher);


        // std::cout<<"\n step1"<<" left "<<idx_left_sub_tree<<" right "<<idx_right_sub_tree;

        // Arguments for_left_sub_tree(n + 1, 2 * l, max_depth, idx_left_sub_tree, partition_color, actual_max_depth, tiling_height);
        // Arguments for_right_sub_tree(n + 1, 2 * l + 1, max_depth, idx_right_sub_tree, partition_color, actual_max_depth, tiling_height);

        // arg_map.set_point(left_sub_tree_color, TaskArgument(&for_left_sub_tree, sizeof(Arguments)));
        // arg_map.set_point(right_sub_tree_color, TaskArgument(&for_right_sub_tree, sizeof(Arguments)));

        // IndexTaskLauncher dummy_compress_launcher(DUMMY_COMPRESS_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        // RegionRequirement req(lp, 0, READ_WRITE, EXCLUSIVE, lr);
        // req.add_field(FID_X);
        // dummy_compress_launcher.add_region_requirement(req);
        // std::cout<<"\n step2";
        // runtime->execute_index_space(ctxt, dummy_compress_launcher);

    // } else if(n == tiling_height) {
        // int new_val = static_cast<coord_t>(pow(2, max_depth-(n+2))) + 1;
        // idx_left_sub_tree = idx + static_cast<coord_t>(pow(2, n)) - l + (2 * l * new_val);
        // idx_right_sub_tree = idx_left_sub_tree + new_val;

        // Rect<1> launch_domain(left_sub_tree_color, right_sub_tree_color);
        // ArgumentMap arg_map;

        // Arguments for_left_sub_tree(n + 1, 2 * l, max_depth, idx_left_sub_tree, partition_color, actual_max_depth, tiling_height);
        // Arguments for_right_sub_tree(n + 1, 2 * l + 1, max_depth, idx_right_sub_tree, partition_color, actual_max_depth, tiling_height);

        // arg_map.set_point(left_sub_tree_color, TaskArgument(&for_left_sub_tree, sizeof(Arguments)));
        // arg_map.set_point(right_sub_tree_color, TaskArgument(&for_right_sub_tree, sizeof(Arguments)));

        // IndexTaskLauncher compress_task(COMPRESS_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        // RegionRequirement req(lp, 0, READ_WRITE, EXCLUSIVE, lr);
        // req.add_field(FID_X);
        // compress_task.add_region_requirement(req);
        // runtime->execute_index_space(ctxt, compress_task);

    // idx_left_sub_tree = 7;
    // idx_right_sub_tree = 10;

    // Rect<1> launch_domain(Point<1>(0LL), Point<1>(8LL));
    ArgumentMap arg_map;

    Arguments for_first_sub_tree(3, 2 * l, max_depth, 7, partition_color, actual_max_depth, tiling_height);
    Arguments for_second_sub_tree(3, 2 * l, max_depth, 10, partition_color, actual_max_depth, tiling_height);
    Arguments for_third_sub_tree(3, 2 * l, max_depth, 13, partition_color, actual_max_depth, tiling_height);
    Arguments for_fourth_sub_tree(3, 2 * l, max_depth, 16, partition_color, actual_max_depth, tiling_height);
    Arguments for_fifth_sub_tree(3, 2 * l, max_depth, 19, partition_color, actual_max_depth, tiling_height);
    Arguments for_sixth_sub_tree(3, 2 * l, max_depth, 22, partition_color, actual_max_depth, tiling_height);
    Arguments for_seventh_sub_tree(3, 2 * l, max_depth, 25, partition_color, actual_max_depth, tiling_height);
    Arguments for_eighth_sub_tree(3, 2 * l, max_depth, 28, partition_color, actual_max_depth, tiling_height);

    TaskLauncher compress_launcher(COMPRESS_TASK_ID, TaskArgument(&for_first_sub_tree, sizeof(Arguments)));
    RegionRequirement req(first_sub_tree_lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(FID_X);
    compress_launcher.add_region_requirement(req);
    std::cout<<"\n step2";
    runtime->execute_task(ctxt, compress_launcher);


    // }
}


std::vector<ReturnGetCoefArguments> path;

struct ReturnGetCoefArguments get_coef_util_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctxt, HighLevelRuntime *runtime) {
    GetCoefUtilArguments args = task->is_index_space ? *(const GetCoefUtilArguments *) task->local_args
    : *(const GetCoefUtilArguments *) task->args;

    int n = args.n;
    int l = args.l;
    int max_depth = args.max_depth;
    int questioned_n = args.questioned_n;
    int questioned_l = args.questioned_l;

    DomainPoint left_sub_tree_color(Point<1>(1LL));
    DomainPoint right_sub_tree_color(Point<1>(2LL));
    Color partition_color = args.partition_color;

    coord_t idx = args.idx;

    assert(regions.size() == 1);
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalPartition lp = LogicalPartition::NO_PART;

    lp = runtime->get_logical_partition_by_color(ctxt, lr, partition_color);

    LogicalRegion left_sub_tree_lr, right_sub_tree_lr;

    ReturnGetCoefArguments get_coef_args(n, l, lr, idx, true);
    path.push_back(get_coef_args);

    left_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, left_sub_tree_color);
    right_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, right_sub_tree_color);

    coord_t idx_left_sub_tree = 0LL;
    coord_t idx_right_sub_tree = 0LL;

    IndexSpace indexspace_left = left_sub_tree_lr.get_index_space();
    IndexSpace indexspace_right = right_sub_tree_lr.get_index_space();

    if(n == questioned_n && l == questioned_l) {
        ReturnGetCoefArguments get_coef_args(n, l, lr, idx, true);
        return get_coef_args;
    }

    Future f_left = Future::from_value(runtime, -1), f_right = Future::from_value(runtime, -1);

    bool left_partition = false, right_partition = false;

    struct ReturnGetCoefArguments parent = args.parent;

    if (runtime->has_index_partition(ctxt, indexspace_left, partition_color)) {
        idx_left_sub_tree = idx + 1;

        parent.n = n;
        parent.l = l;
        parent.lr = lr;
        parent.idx = idx;
        parent.exists = false;

        GetCoefUtilArguments for_left_sub_tree(n + 1, l * 2, max_depth, idx_left_sub_tree, partition_color, questioned_n, questioned_l, parent);

        TaskLauncher get_coefs_launcher(GET_COEF_UTIL_TASK_ID, TaskArgument(&for_left_sub_tree, sizeof(GetCoefUtilArguments)));
        RegionRequirement req(left_sub_tree_lr, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(FID_X);
        get_coefs_launcher.add_region_requirement(req);
        f_left = runtime->execute_task(ctxt, get_coefs_launcher);
        left_partition = true;
    }

    if (runtime->has_index_partition(ctxt, indexspace_right, partition_color)) {
        idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));

        parent.n = n;
        parent.l = l;
        parent.lr = lr;
        parent.idx = idx;
        parent.exists = false;

        GetCoefUtilArguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, partition_color, questioned_n, questioned_l, parent);

        TaskLauncher get_coefs_launcher(GET_COEF_UTIL_TASK_ID, TaskArgument(&for_right_sub_tree, sizeof(GetCoefUtilArguments)));
        RegionRequirement req(right_sub_tree_lr, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(FID_X);
        get_coefs_launcher.add_region_requirement(req);
        f_right = runtime->execute_task(ctxt, get_coefs_launcher);
        right_partition = true;
    }

    if(left_partition || right_partition) {
        if (f_left.get_result<ReturnGetCoefArguments>().lr != LogicalRegion::NO_REGION) {
            return f_left.get_result<ReturnGetCoefArguments>();
        } else {
            return f_right.get_result<ReturnGetCoefArguments>();
        }
    } else if(n == max_depth - 1){
        while(questioned_n >= 0) {
            for (int i = path.size() - 1; i >= 0; i--) {
                if (path[i].n == (questioned_n - 1) && path[i].l == questioned_l/2) {
                    ReturnGetCoefArguments get_coef_args(path[i].n, path[i].l, path[i].lr, path[i].idx, false);
                    return get_coef_args;                 
                }
            }
            questioned_n--;
            questioned_l = questioned_l/2;
        }
    }

    ReturnGetCoefArguments get_coef_args1(-1, -1, LogicalRegion::NO_REGION, -1,false);
    return get_coef_args1; 
}

int get_coef_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctxt, HighLevelRuntime *runtime) {
    GetCoefArguments args = task->is_index_space ? *(const GetCoefArguments *) task->local_args
    : *(const GetCoefArguments *) task->args;

    int n = args.n;
    int max_depth = args.max_depth;
    int questioned_n = args.questioned_n;
    int questioned_l = args.questioned_l;
    Color partition_color = args.partition_color; 

    DomainPoint left_sub_tree_color(Point<1>(1LL));
    DomainPoint right_sub_tree_color(Point<1>(2LL));

    assert(regions.size() == 1);
    LogicalRegion lr = regions[0].get_logical_region();

    int val = pow(2, n);

    if(questioned_l < 0 || questioned_l >= val) {
        return 0;
    }

    ReturnGetCoefArguments parent(-1, -1, LogicalRegion::NO_REGION, -1,false);

    GetCoefUtilArguments get_coef_args(0, 0, max_depth, 0, partition_color, questioned_n, questioned_l, parent);
    TaskLauncher get_coefs_util_launcher(GET_COEF_UTIL_TASK_ID, TaskArgument(&get_coef_args, sizeof(GetCoefUtilArguments)));
    get_coefs_util_launcher.add_region_requirement(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
    get_coefs_util_launcher.add_field(0, FID_X);
    Future return_coeficient = runtime->execute_task(ctxt, get_coefs_util_launcher);

    ReturnGetCoefArguments return_coef = return_coeficient.get_result<ReturnGetCoefArguments>();

    if (return_coef.n == -1) {
        return 0;
    }

    LogicalPartition lp = runtime->get_logical_partition_by_color(ctxt, return_coef.lr, partition_color);

    LogicalRegion left_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, left_sub_tree_color);
    LogicalRegion right_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, right_sub_tree_color);

    IndexSpace indexspace_left = left_sub_tree_lr.get_index_space();
    IndexSpace indexspace_right = right_sub_tree_lr.get_index_space();

    if (runtime->has_index_partition(ctxt, indexspace_left, partition_color) == true &&
        runtime->has_index_partition(ctxt, indexspace_right, partition_color) == true && return_coef.exists == true) {
        return -1;
    }

    Future f1;
    if(return_coef.lr != LogicalRegion::NO_REGION) {
        int index = return_coef.idx;
        {
            ReadTaskArgs args(index);
            TaskLauncher read_task_launcher(READ_TASK_ID, TaskArgument(&args, sizeof(ReadTaskArgs)));
            RegionRequirement req(return_coef.lr, READ_ONLY, EXCLUSIVE, lr);
            req.add_field(FID_X);
            read_task_launcher.add_region_requirement(req);
            f1 = runtime->execute_task(ctxt, read_task_launcher);
        }
        if(return_coef.exists == false) {
            return f1.get_result<int>() +  (2 * (n - return_coef.n));
        } else{
            return f1.get_result<int>();
        }
        
    }

    // fprintf(stderr, "Somthings wrong\n");
    return -1;
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

    coord_t idx = args.idx;

    assert(regions.size() == 4);
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalRegion lr2 = regions[1].get_logical_region();
    LogicalRegion lr_whole = regions[2].get_logical_region();
    LogicalRegion dummy_lr = regions[3].get_logical_region();
    LogicalPartition lp = LogicalPartition::NO_PART, lp2 = LogicalPartition::NO_PART, lp11, lp21;

    IndexSpace indexspace_left = IndexSpace::NO_SPACE, indexspace_right = IndexSpace::NO_SPACE;
    LogicalRegion my_sub_tree_lr = dummy_lr;
    LogicalRegion left_sub_tree_lr = dummy_lr;
    LogicalRegion right_sub_tree_lr = dummy_lr;
    LogicalRegion my_sub_tree_lr2 = LogicalRegion::NO_REGION;
    LogicalRegion left_sub_tree_lr2 = LogicalRegion::NO_REGION;
    LogicalRegion right_sub_tree_lr2 = LogicalRegion::NO_REGION;

    if (lr != dummy_lr) {
        lp = runtime->get_logical_partition_by_color(ctx, lr, partition_color1);
        my_sub_tree_lr = runtime->get_logical_subregion_by_color(ctx, lp, my_sub_tree_color);
        left_sub_tree_lr = runtime->get_logical_subregion_by_color(ctx, lp, left_sub_tree_color);
        right_sub_tree_lr = runtime->get_logical_subregion_by_color(ctx, lp, right_sub_tree_color);

        indexspace_left = left_sub_tree_lr.get_index_space();
        indexspace_right = right_sub_tree_lr.get_index_space();

        assert(lp != LogicalPartition::NO_PART);
    }

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

    if (is_s0_valid == false) {

        bool left_subtree = false, right_subtree = false;

        assert(my_sub_tree_lr2 != LogicalRegion::NO_REGION);
        assert(lr2 != LogicalRegion::NO_REGION);
        assert(lr != LogicalRegion::NO_REGION);
        assert(lr_whole != LogicalRegion::NO_REGION);
        assert(lp != LogicalPartition::NO_PART);
        assert(lp2 != LogicalPartition::NO_PART);


        if (indexspace_left != IndexSpace::NO_SPACE && runtime->has_index_partition(ctx, indexspace_left, partition_color1)) {

            {
                DiffSetTaskArgs args(idx, 0);
                TaskLauncher diff_set_task_launcher(DIFF_SET_TASK_ID, TaskArgument(&args, sizeof(DiffSetTaskArgs)));
                RegionRequirement req(my_sub_tree_lr2, WRITE_DISCARD, EXCLUSIVE, lr2);
                req.add_field(FID_X);
                diff_set_task_launcher.add_region_requirement(req);
                runtime->execute_task(ctx, diff_set_task_launcher);
            }
            DiffArguments for_left_sub_tree (n + 1, l * 2, max_depth, idx_left_sub_tree, partition_color1, partition_color2, actual_max_depth, RANDOM, false);

            TaskLauncher diff_launcher(DIFF_TASK_ID, TaskArgument(&for_left_sub_tree, sizeof(DiffArguments)));
            RegionRequirement req(left_sub_tree_lr, READ_ONLY, EXCLUSIVE, lr);
            RegionRequirement req2(left_sub_tree_lr2, WRITE_DISCARD, EXCLUSIVE, lr2);
            RegionRequirement req3(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole);
            RegionRequirement req4(dummy_lr, READ_ONLY, EXCLUSIVE, dummy_lr);
            req.add_field(FID_X);
            req2.add_field(FID_X);
            req3.add_field(FID_X);
            req4.add_field(FID_X);
            diff_launcher.add_region_requirement(req);
            diff_launcher.add_region_requirement(req2);
            diff_launcher.add_region_requirement(req3);
            diff_launcher.add_region_requirement(req4);
            runtime->execute_task(ctx, diff_launcher);

            left_subtree = true;
        }
        if (indexspace_right != IndexSpace::NO_SPACE && runtime->has_index_partition(ctx, indexspace_right, partition_color1)) {
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
            RegionRequirement req4(dummy_lr, READ_ONLY, EXCLUSIVE, dummy_lr);
            req.add_field(FID_X);
            req2.add_field(FID_X);
            req3.add_field(FID_X);
            req4.add_field(FID_X);
            diff_launcher.add_region_requirement(req);
            diff_launcher.add_region_requirement(req2);
            diff_launcher.add_region_requirement(req3);
            diff_launcher.add_region_requirement(req4);
            runtime->execute_task(ctx, diff_launcher);

            right_subtree = true;
        }

        if (!left_subtree && !right_subtree) {
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

            GetCoefArguments get_coef_args_sm(n, l, max_depth, 0, partition_color1, n, l - 1);
            Future f_sm;
            {
                TaskLauncher get_coefs_launcher(GET_COEF_TASK_ID, TaskArgument(&get_coef_args_sm, sizeof(GetCoefArguments)));
                get_coefs_launcher.add_region_requirement(RegionRequirement(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole));
                get_coefs_launcher.add_field(0, FID_X);
                f_sm = runtime->execute_task(ctx, get_coefs_launcher);
            }
            sm = f_sm.get_result<int>();

            GetCoefArguments get_coef_args_sp(n, l, max_depth, 0, partition_color1, n, l + 1);
            
            Future f_sp;
            {
                TaskLauncher get_coefs_launcher(GET_COEF_TASK_ID, TaskArgument(&get_coef_args_sp, sizeof(GetCoefArguments)));
                get_coefs_launcher.add_region_requirement(RegionRequirement(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole));
                get_coefs_launcher.add_field(0, FID_X);
                f_sp = runtime->execute_task(ctx, get_coefs_launcher);
            }
            sp = f_sp.get_result<int>();

            r = 0;
            bool if_is_true = false;

            if (sm >= 0 && sp >= 0 && s0 >= 0) {
                r = sm + sp + s0;
                if_is_true = true;
            }

            {
                DiffSetTaskArgs args(idx, r);
                TaskLauncher diff_set_task_launcher(DIFF_SET_TASK_ID, TaskArgument(&args, sizeof(DiffSetTaskArgs)));
                RegionRequirement req(my_sub_tree_lr2, WRITE_DISCARD, EXCLUSIVE, lr2);
                req.add_field(FID_X);
                diff_set_task_launcher.add_region_requirement(req);
                runtime->execute_task(ctx, diff_set_task_launcher);
            }

            if (if_is_true == false) {
                DiffArguments for_left_sub_tree (n + 1, l * 2, max_depth, idx_left_sub_tree, partition_color1, partition_color2, actual_max_depth, ceil(s0/float(2)), true);
                DiffArguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, partition_color1, partition_color2, actual_max_depth, ceil(s0/float(2)), true);

                TaskLauncher diff_launcher_left(DIFF_TASK_ID, TaskArgument(&for_left_sub_tree, sizeof(DiffArguments)));
                RegionRequirement req_left(dummy_lr, READ_ONLY, EXCLUSIVE, dummy_lr);
                RegionRequirement req_left2(left_sub_tree_lr2, WRITE_DISCARD, EXCLUSIVE, lr2);
                RegionRequirement req_left3(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole);
                RegionRequirement req_left4(dummy_lr, READ_ONLY, EXCLUSIVE, dummy_lr);
                req_left.add_field(FID_X);
                req_left2.add_field(FID_X);
                req_left3.add_field(FID_X);
                req_left4.add_field(FID_X);
                diff_launcher_left.add_region_requirement(req_left);
                diff_launcher_left.add_region_requirement(req_left2);
                diff_launcher_left.add_region_requirement(req_left3);
                diff_launcher_left.add_region_requirement(req_left4);
                runtime->execute_task(ctx, diff_launcher_left);

                TaskLauncher diff_launcher_right(DIFF_TASK_ID, TaskArgument(&for_right_sub_tree, sizeof(DiffArguments)));
                RegionRequirement req_right(dummy_lr, READ_ONLY, EXCLUSIVE, dummy_lr);
                RegionRequirement req_right2(right_sub_tree_lr2, WRITE_DISCARD, EXCLUSIVE, lr2);
                RegionRequirement req_right3(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole);
                RegionRequirement req_right4(dummy_lr, READ_ONLY, EXCLUSIVE, dummy_lr);
                req_right.add_field(FID_X);
                req_right2.add_field(FID_X);
                req_right3.add_field(FID_X);
                req_right4.add_field(FID_X);
                diff_launcher_right.add_region_requirement(req_right);
                diff_launcher_right.add_region_requirement(req_right2);
                diff_launcher_right.add_region_requirement(req_right3);
                diff_launcher_right.add_region_requirement(req_right4);
                runtime->execute_task(ctx, diff_launcher_right);

            }
        }
    } else {
        if (l % 2 == 0) {

            
            sp = s0;
            GetCoefArguments get_coef_args_sm(n, l, max_depth, 0, partition_color1, n, l - 1);
            Future f_sm;
            {
                TaskLauncher get_coefs_launcher(GET_COEF_TASK_ID, TaskArgument(&get_coef_args_sm, sizeof(GetCoefArguments)));
                get_coefs_launcher.add_region_requirement(RegionRequirement(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole));
                get_coefs_launcher.add_field(0, FID_X);
                f_sm = runtime->execute_task(ctx, get_coefs_launcher);
            }
            sm = f_sm.get_result<int>();
        } else {
            sm = s0;
            GetCoefArguments get_coef_args_sp(n, l, max_depth, 0, partition_color1, n, l + 1);
            Future f_sp;
            {
                TaskLauncher get_coefs_launcher(GET_COEF_TASK_ID, TaskArgument(&get_coef_args_sp, sizeof(GetCoefArguments)));
                get_coefs_launcher.add_region_requirement(RegionRequirement(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole));
                get_coefs_launcher.add_field(0, FID_X);
                f_sp = runtime->execute_task(ctx, get_coefs_launcher);
            }
            sp = f_sp.get_result<int>();
        }

        r = 0;

        bool if_is_true = false;
        if (sm >= 0 && sp >= 0 && s0 >= 0) {
            r = sm + sp + s0;
            if_is_true = true;
        }

        {
            DiffSetTaskArgs args(idx, r);
            TaskLauncher diff_set_task_launcher(DIFF_SET_TASK_ID, TaskArgument(&args, sizeof(DiffSetTaskArgs)));
            RegionRequirement req(my_sub_tree_lr2, WRITE_DISCARD, EXCLUSIVE, lr2);
            req.add_field(FID_X);
            diff_set_task_launcher.add_region_requirement(req);
            runtime->execute_task(ctx, diff_set_task_launcher);
        }


        if (if_is_true == false) {
            DiffArguments for_left_sub_tree (n + 1, l * 2    , max_depth, idx_left_sub_tree, partition_color1, partition_color2, actual_max_depth, ceil(s0/float(2)), true);
            DiffArguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, partition_color1, partition_color2, actual_max_depth, ceil(s0/float(2)), true);

            TaskLauncher diff_launcher_left(DIFF_TASK_ID, TaskArgument(&for_left_sub_tree, sizeof(DiffArguments)));
            RegionRequirement req_left(left_sub_tree_lr, READ_ONLY, EXCLUSIVE, lr);
            RegionRequirement req_left2(left_sub_tree_lr2, WRITE_DISCARD, EXCLUSIVE, lr2);
            RegionRequirement req_left3(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole);
            RegionRequirement req_left4(dummy_lr, READ_ONLY, EXCLUSIVE, dummy_lr);
            req_left.add_field(FID_X);
            req_left2.add_field(FID_X);
            req_left3.add_field(FID_X);
            req_left4.add_field(FID_X);
            diff_launcher_left.add_region_requirement(req_left);
            diff_launcher_left.add_region_requirement(req_left2);
            diff_launcher_left.add_region_requirement(req_left3);
            diff_launcher_left.add_region_requirement(req_left4);
            runtime->execute_task(ctx, diff_launcher_left);

            TaskLauncher diff_launcher_right(DIFF_TASK_ID, TaskArgument(&for_right_sub_tree, sizeof(DiffArguments)));
            RegionRequirement req_right(right_sub_tree_lr, READ_ONLY, EXCLUSIVE, lr);
            RegionRequirement req_right2(right_sub_tree_lr2, WRITE_DISCARD, EXCLUSIVE, lr2);
            RegionRequirement req_right3(lr_whole, READ_ONLY, EXCLUSIVE, lr_whole);
            RegionRequirement req_right4(dummy_lr, READ_ONLY, EXCLUSIVE, dummy_lr);
            req_right.add_field(FID_X);
            req_right2.add_field(FID_X);
            req_right3.add_field(FID_X);
            req_right4.add_field(FID_X);
            diff_launcher_right.add_region_requirement(req_right);
            diff_launcher_right.add_region_requirement(req_right2);
            diff_launcher_right.add_region_requirement(req_right3);
            diff_launcher_right.add_region_requirement(req_right4);
            runtime->execute_task(ctx, diff_launcher_right);
        }

    }
}

int inner_product_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {
    InnerProductArguments args = task->is_index_space ? *(const InnerProductArguments *) task->local_args
    : *(const InnerProductArguments *) task->args;

    int n = args.n;
    int l = args.l;
    int max_depth = args.max_depth;
    int actual_max_depth = args.actual_max_depth;

    DomainPoint my_sub_tree_color(Point<1>(0LL));
    DomainPoint left_sub_tree_color(Point<1>(1LL));
    DomainPoint right_sub_tree_color(Point<1>(2LL));
    Color partition_color1 = args.partition_color1;
    Color partition_color2 = args.partition_color2;

    coord_t idx_left_sub_tree = 0LL;
    coord_t idx_right_sub_tree = 0LL;
 
    coord_t idx = args.idx;

    assert(regions.size() == 2);

    LogicalRegion lr1 = regions[0].get_logical_region();
    LogicalRegion lr2 = regions[1].get_logical_region();

    Domain left_tree = runtime->get_index_space_domain(ctx, lr1.get_index_space());
    Domain right_tree = runtime->get_index_space_domain(ctx, lr2.get_index_space());

    // To compare so that both the trees have same layout structure
    assert(left_tree == right_tree);
    
    LogicalPartition lp1 = LogicalPartition::NO_PART, lp2 = LogicalPartition::NO_PART;

    LogicalRegion my_sub_tree_lr1 = LogicalRegion::NO_REGION;
    LogicalRegion left_sub_tree_lr1 = LogicalRegion::NO_REGION;
    LogicalRegion right_sub_tree_lr1 = LogicalRegion::NO_REGION;
    LogicalRegion my_sub_tree_lr2 = LogicalRegion::NO_REGION;
    LogicalRegion left_sub_tree_lr2 = LogicalRegion::NO_REGION;
    LogicalRegion right_sub_tree_lr2 = LogicalRegion::NO_REGION;

    IndexSpace indexspace_tree_left1 = IndexSpace::NO_SPACE, indexspace_tree_left2 = IndexSpace::NO_SPACE;
    IndexSpace indexspace_tree_right1 = IndexSpace::NO_SPACE, indexspace_tree_right2 = IndexSpace::NO_SPACE;

    if (n >= actual_max_depth)
        return 0;

    lp1 = runtime->get_logical_partition_by_color(ctx, lr1, partition_color1);
    my_sub_tree_lr1 = runtime->get_logical_subregion_by_color(ctx, lp1, my_sub_tree_color);
    left_sub_tree_lr1 = runtime->get_logical_subregion_by_color(ctx, lp1, left_sub_tree_color);
    right_sub_tree_lr1 = runtime->get_logical_subregion_by_color(ctx, lp1, right_sub_tree_color);

    lp2 = runtime->get_logical_partition_by_color(ctx, lr2, partition_color2);
    my_sub_tree_lr2 = runtime->get_logical_subregion_by_color(ctx, lp2, my_sub_tree_color);
    left_sub_tree_lr2 = runtime->get_logical_subregion_by_color(ctx, lp2, left_sub_tree_color);
    right_sub_tree_lr2 = runtime->get_logical_subregion_by_color(ctx, lp2, right_sub_tree_color);

    indexspace_tree_left1 = left_sub_tree_lr1.get_index_space();
    indexspace_tree_left2 = left_sub_tree_lr2.get_index_space();

    indexspace_tree_right1 = right_sub_tree_lr1.get_index_space();
    indexspace_tree_right2 = right_sub_tree_lr2.get_index_space();

    Future f_left;
    {
        ReadTaskArgs args(idx);
        TaskLauncher read_task_launcher(READ_TASK_ID, TaskArgument(&args, sizeof(ReadTaskArgs)));
        RegionRequirement req(my_sub_tree_lr1, READ_ONLY, EXCLUSIVE, lr1);
        req.add_field(FID_X);
        read_task_launcher.add_region_requirement(req);
        f_left = runtime->execute_task(ctx, read_task_launcher);
    }

    Future f_right;
    {
        ReadTaskArgs args(idx);
        TaskLauncher read_task_launcher(READ_TASK_ID, TaskArgument(&args, sizeof(ReadTaskArgs)));
        RegionRequirement req(my_sub_tree_lr2, READ_ONLY, EXCLUSIVE, lr2);
        req.add_field(FID_X);
        read_task_launcher.add_region_requirement(req);
        f_right = runtime->execute_task(ctx, read_task_launcher);
    }

    Future f_result_left = Future::from_value(runtime, 0), f_result_right = Future::from_value(runtime, 0);

    if ((indexspace_tree_left1 != IndexSpace::NO_SPACE && runtime->has_index_partition(ctx, indexspace_tree_left1, partition_color1)) && 
        (indexspace_tree_left2 != IndexSpace::NO_SPACE && runtime->has_index_partition(ctx, indexspace_tree_left2, partition_color2)) ) {

        idx_left_sub_tree = idx + 1;
        assert(lp2 != LogicalPartition::NO_PART);
        assert(lp1 != LogicalPartition::NO_PART);
        InnerProductArguments for_left_sub_tree (n + 1, l * 2, max_depth, idx_left_sub_tree, partition_color1, partition_color2, actual_max_depth);

        TaskLauncher inner_product_launcher(INNER_PRODUCT_TASK_ID, TaskArgument(&for_left_sub_tree, sizeof(InnerProductArguments)));
        RegionRequirement req1(lr1, READ_ONLY, EXCLUSIVE, lr1);
        RegionRequirement req2(lr2, READ_ONLY, EXCLUSIVE, lr2);
        req1.add_field(FID_X);
        req2.add_field(FID_X);
        inner_product_launcher.add_region_requirement(req1);
        inner_product_launcher.add_region_requirement(req2);

        f_result_left = runtime->execute_task(ctx, inner_product_launcher);
    }

    if ((indexspace_tree_right1 != IndexSpace::NO_SPACE && runtime->has_index_partition(ctx, indexspace_tree_right1, partition_color1)) && 
        (indexspace_tree_right2 != IndexSpace::NO_SPACE && runtime->has_index_partition(ctx, indexspace_tree_right2, partition_color2)) ) {

        idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));
        assert(lp2 != LogicalPartition::NO_PART);
        assert(lp1 != LogicalPartition::NO_PART);
        InnerProductArguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, partition_color1, partition_color2, actual_max_depth);

        TaskLauncher inner_product_launcher(INNER_PRODUCT_TASK_ID, TaskArgument(&for_right_sub_tree, sizeof(InnerProductArguments)));
        RegionRequirement req1(lr1, READ_ONLY, EXCLUSIVE, lr1);
        RegionRequirement req2(lr2, READ_ONLY, EXCLUSIVE, lr2);
        req1.add_field(FID_X);
        req2.add_field(FID_X);
        inner_product_launcher.add_region_requirement(req1);
        inner_product_launcher.add_region_requirement(req2);

        f_result_right = runtime->execute_task(ctx, inner_product_launcher);
    }

    TaskLauncher product_task_launcher(PRODUCT_TASK_ID, TaskArgument(NULL, 0));
    product_task_launcher.add_future(f_left);
    product_task_launcher.add_future(f_right);
    product_task_launcher.add_future(f_result_left);
    product_task_launcher.add_future(f_result_right);
    Future result = runtime->execute_task(ctx, product_task_launcher);

    return result.get_result<int>();
}

int product_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime) {
  assert(task->futures.size() == 4);
  Future f_left = task->futures[0];
  int r_left = f_left.get_result<int>();
  Future f_right = task->futures[1];
  int r_right = f_right.get_result<int>();
  Future f_result_left = task->futures[2];
  int r_result_left = f_result_left.get_result<int>();
  Future f_result_right = task->futures[3];
  int r_result_right = f_result_right.get_result<int>();

  return ((r_left * r_right) + r_result_left + r_result_right);
}

void gaxpy_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {
    GaxpyArguments args = task->is_index_space ? *(const GaxpyArguments *) task->local_args
    : *(const GaxpyArguments *) task->args;

    int n = args.n;
    int l = args.l;
    int max_depth = args.max_depth;
    int actual_max_depth = args.actual_max_depth;

    coord_t idx = args.idx;

    DomainPoint my_sub_tree_color(Point<1>(0LL));
    DomainPoint left_sub_tree_color(Point<1>(1LL));
    DomainPoint right_sub_tree_color(Point<1>(2LL));
    Color partition_color1 = args.partition_color1;
    Color partition_color2 = args.partition_color2;
    Color partition_color3 = args.partition_color3;
    int left_tree_depth = args.left_tree_depth;
    int right_tree_depth = args.right_tree_depth;

    coord_t idx_left_sub_tree = 0LL;
    coord_t idx_right_sub_tree = 0LL;

    assert(regions.size() == 4);

    bool is_left = true, is_right = true;

    LogicalRegion lr1 = regions[0].get_logical_region();
    LogicalRegion lr2 = regions[1].get_logical_region();
    LogicalRegion lr3 = regions[2].get_logical_region();
    LogicalRegion dummy_lr = regions[3].get_logical_region();

    assert(lr1 != LogicalRegion::NO_REGION);
    assert(lr2 != LogicalRegion::NO_REGION);
    assert(lr3 != LogicalRegion::NO_REGION);

    if (lr1 != dummy_lr && lr2 != dummy_lr) {
        Domain left_tree = runtime->get_index_space_domain(ctx, lr1.get_index_space());
        Domain right_tree = runtime->get_index_space_domain(ctx, lr2.get_index_space());
        // To compare so that both the trees have same layout structure
        assert(left_tree == right_tree);
    }

    LogicalPartition lp1 = LogicalPartition::NO_PART, lp2 = LogicalPartition::NO_PART, lp3 = LogicalPartition::NO_PART, dummy_lp;

    idx_left_sub_tree = idx + 1;
    idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));

    LogicalRegion my_sub_tree_lr1 = LogicalRegion::NO_REGION;
    LogicalRegion left_sub_tree_lr1 = LogicalRegion::NO_REGION;
    LogicalRegion right_sub_tree_lr1 = LogicalRegion::NO_REGION;
    LogicalRegion my_sub_tree_lr2 = LogicalRegion::NO_REGION;
    LogicalRegion left_sub_tree_lr2 = LogicalRegion::NO_REGION;
    LogicalRegion right_sub_tree_lr2 = LogicalRegion::NO_REGION;
    LogicalRegion my_sub_tree_lr3 = lr3, left_sub_tree_lr3, right_sub_tree_lr3;

    IndexSpace indexspace_tree_left1 = IndexSpace::NO_SPACE, indexspace_tree_left2 = IndexSpace::NO_SPACE;
    IndexSpace indexspace_tree_right1 = IndexSpace::NO_SPACE, indexspace_tree_right2 = IndexSpace::NO_SPACE;
    
    bool left_subtree = false, right_subtree = false;

    if (lr1 != dummy_lr && lr1 != LogicalRegion::NO_REGION && runtime->has_logical_partition_by_color(ctx, lr1, partition_color1)) {
        lp1 = runtime->get_logical_partition_by_color(ctx, lr1, partition_color1);
        my_sub_tree_lr1 = runtime->get_logical_subregion_by_color(ctx, lp1, my_sub_tree_color);
        left_sub_tree_lr1 = runtime->get_logical_subregion_by_color(ctx, lp1, left_sub_tree_color);
        right_sub_tree_lr1 = runtime->get_logical_subregion_by_color(ctx, lp1, right_sub_tree_color);
        left_subtree = true;
    }
    if(n == left_tree_depth - 1){
        lp1 = runtime->get_logical_partition_by_color(ctx, lr1, partition_color1);
        my_sub_tree_lr1 = runtime->get_logical_subregion_by_color(ctx, lp1, my_sub_tree_color);
    }

    if (lr2 != dummy_lr && lr2 != LogicalRegion::NO_REGION && runtime->has_logical_partition_by_color(ctx, lr2, partition_color2)) {
        lp2 = runtime->get_logical_partition_by_color(ctx, lr2, partition_color2);
        my_sub_tree_lr2 = runtime->get_logical_subregion_by_color(ctx, lp2, my_sub_tree_color);
        left_sub_tree_lr2 = runtime->get_logical_subregion_by_color(ctx, lp2, left_sub_tree_color);
        right_sub_tree_lr2 = runtime->get_logical_subregion_by_color(ctx, lp2, right_sub_tree_color);
        right_subtree = true;
    }

    if(n == right_tree_depth - 1){
        lp2 = runtime->get_logical_partition_by_color(ctx, lr2, partition_color2);
        my_sub_tree_lr2 = runtime->get_logical_subregion_by_color(ctx, lp2, my_sub_tree_color);
    }

    if ((left_subtree || right_subtree) && n < actual_max_depth) { // TODO: probably this can be removed 
        IndexSpace is3 = lr3.get_index_space();
        DomainPointColoring coloring;

        Rect<1> my_sub_tree_rect(idx, idx);
        Rect<1> left_sub_tree_rect(idx_left_sub_tree, idx_right_sub_tree - 1);
        Rect<1> right_sub_tree_rect(idx_right_sub_tree,
        idx_right_sub_tree + static_cast<coord_t>(pow(2, max_depth - n)) - 2);

        coloring[my_sub_tree_color] = my_sub_tree_rect;
        coloring[left_sub_tree_color] = left_sub_tree_rect;
        coloring[right_sub_tree_color] = right_sub_tree_rect;

        Rect<1> color_space = Rect<1>(my_sub_tree_color, right_sub_tree_color);

        IndexPartition ip = runtime->create_index_partition(ctx, is3, color_space, coloring, DISJOINT_KIND, partition_color3);
        lp3 = runtime->get_logical_partition(ctx, lr3, ip);
        my_sub_tree_lr3 = runtime->get_logical_subregion_by_color(ctx, lp3, my_sub_tree_color);
        left_sub_tree_lr3 = runtime->get_logical_subregion_by_color(ctx, lp3, left_sub_tree_color);
        right_sub_tree_lr3 = runtime->get_logical_subregion_by_color(ctx, lp3, right_sub_tree_color);
    } else {
        return;
    }

    assert(lr3 != LogicalRegion::NO_REGION);
    assert(my_sub_tree_lr3 != LogicalRegion::NO_REGION);

    if (lr1 != LogicalRegion::NO_REGION) {
        indexspace_tree_left1 = left_sub_tree_lr1.get_index_space();
        indexspace_tree_right1 = right_sub_tree_lr1.get_index_space();
    }
    if (lr2 != LogicalRegion::NO_REGION) {
        indexspace_tree_left2 = left_sub_tree_lr2.get_index_space();
        indexspace_tree_right2 = right_sub_tree_lr2.get_index_space();
    }

    if ((indexspace_tree_left1 != IndexSpace::NO_SPACE && runtime->has_index_partition(ctx, indexspace_tree_left1, partition_color1)) || 
        (indexspace_tree_left2 != IndexSpace::NO_SPACE && runtime->has_index_partition(ctx, indexspace_tree_left2, partition_color2)) ) {

        {
            // when left tree has reached its leaf node
            if (n != left_tree_depth - 1 && (indexspace_tree_left1 == IndexSpace::NO_SPACE || runtime->has_index_partition(ctx, indexspace_tree_left1, partition_color1) == false)) {
                is_left = false;
                is_right = true;
                my_sub_tree_lr1 = dummy_lr;
                left_sub_tree_lr1 = dummy_lr;
            }

            // when right tree has reached its leaf node
            if (n != right_tree_depth - 1 && (indexspace_tree_left2 == IndexSpace::NO_SPACE || runtime->has_index_partition(ctx, indexspace_tree_left2, partition_color2) == false)) {
                is_left = true;
                is_right = false;
                my_sub_tree_lr2 = dummy_lr;
                left_sub_tree_lr2 = dummy_lr;
            }

        }

        assert(left_sub_tree_lr2 != LogicalRegion::NO_REGION);
        assert(left_sub_tree_lr1 != LogicalRegion::NO_REGION);
        assert(left_sub_tree_lr3 != LogicalRegion::NO_REGION);

        GaxpyArguments for_left_sub_tree (n + 1, l * 2, max_depth, idx_left_sub_tree, partition_color1, partition_color2, partition_color3, actual_max_depth, left_tree_depth, right_tree_depth);

        TaskLauncher gaxpy_launcher(GAXPY_TASK_ID, TaskArgument(&for_left_sub_tree, sizeof(GaxpyArguments)));

        if (left_sub_tree_lr1 == dummy_lr) {
            RegionRequirement req1(dummy_lr, READ_ONLY, EXCLUSIVE, dummy_lr);
            req1.add_field(FID_X);
            gaxpy_launcher.add_region_requirement(req1);
        } else {
            RegionRequirement req1(left_sub_tree_lr1, READ_ONLY, EXCLUSIVE, lr1);
            req1.add_field(FID_X);
            gaxpy_launcher.add_region_requirement(req1);
        }

        if (left_sub_tree_lr2 == dummy_lr) {
            RegionRequirement req2(dummy_lr, READ_ONLY, EXCLUSIVE, dummy_lr);
            req2.add_field(FID_X);
            gaxpy_launcher.add_region_requirement(req2);
        } else {
            RegionRequirement req2(left_sub_tree_lr2, READ_ONLY, EXCLUSIVE, lr2);
            req2.add_field(FID_X);
            gaxpy_launcher.add_region_requirement(req2);
        }
        
        RegionRequirement req3(left_sub_tree_lr3, WRITE_DISCARD, EXCLUSIVE, lr3);
        RegionRequirement req4(dummy_lr, READ_ONLY, EXCLUSIVE, dummy_lr);
        req3.add_field(FID_X);
        req4.add_field(FID_X);
        gaxpy_launcher.add_region_requirement(req3);
        gaxpy_launcher.add_region_requirement(req4);
        runtime->execute_task(ctx, gaxpy_launcher);
    }


    if ((indexspace_tree_right1 != IndexSpace::NO_SPACE && runtime->has_index_partition(ctx, indexspace_tree_right1, partition_color1)) || 
        (indexspace_tree_right2 != IndexSpace::NO_SPACE && runtime->has_index_partition(ctx, indexspace_tree_right2, partition_color2)) ) {

        {
            // when left tree has reached its leaf node
            if (n != left_tree_depth - 1 && (indexspace_tree_right1 == IndexSpace::NO_SPACE || runtime->has_index_partition(ctx, indexspace_tree_right1, partition_color1) == false)) {
                is_left = false;
                is_right = true;
                my_sub_tree_lr1 = dummy_lr;
                right_sub_tree_lr1 = dummy_lr;
            }

            // when right tree has reached its leaf node
            if (n != right_tree_depth - 1 && (indexspace_tree_right2 == IndexSpace::NO_SPACE || runtime->has_index_partition(ctx, indexspace_tree_right2, partition_color2) == false)) {
                is_left = true;
                is_right = false;
                my_sub_tree_lr2 = dummy_lr;
                right_sub_tree_lr2 = dummy_lr;
            }
        }

        assert(right_sub_tree_lr2 != LogicalRegion::NO_REGION);
        assert(right_sub_tree_lr1 != LogicalRegion::NO_REGION);
        assert(right_sub_tree_lr3 != LogicalRegion::NO_REGION);

        GaxpyArguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, partition_color1, partition_color2, partition_color3, actual_max_depth, left_tree_depth, right_tree_depth);

        TaskLauncher gaxpy_launcher(GAXPY_TASK_ID, TaskArgument(&for_right_sub_tree, sizeof(GaxpyArguments)));

        if (right_sub_tree_lr1 == dummy_lr) {
            RegionRequirement req1(dummy_lr, READ_ONLY, EXCLUSIVE, dummy_lr);
            req1.add_field(FID_X);
            gaxpy_launcher.add_region_requirement(req1);
        } else {
            RegionRequirement req1(right_sub_tree_lr1, READ_ONLY, EXCLUSIVE, lr1);
            req1.add_field(FID_X);
            gaxpy_launcher.add_region_requirement(req1);
        }

        if (right_sub_tree_lr2 == dummy_lr) {
            RegionRequirement req2(dummy_lr, READ_ONLY, EXCLUSIVE, dummy_lr);
            req2.add_field(FID_X);
            gaxpy_launcher.add_region_requirement(req2);
        } else {
            RegionRequirement req2(right_sub_tree_lr2, READ_ONLY, EXCLUSIVE, lr2);
            req2.add_field(FID_X);
            gaxpy_launcher.add_region_requirement(req2);
        }

        RegionRequirement req3(right_sub_tree_lr3, WRITE_DISCARD, EXCLUSIVE, lr3);
        RegionRequirement req4(dummy_lr, READ_ONLY, EXCLUSIVE, dummy_lr);
        req3.add_field(FID_X);
        req4.add_field(FID_X);
        gaxpy_launcher.add_region_requirement(req3);
        gaxpy_launcher.add_region_requirement(req4);
        runtime->execute_task(ctx, gaxpy_launcher);
    }


    if(my_sub_tree_lr1 == LogicalRegion::NO_REGION)
        my_sub_tree_lr1 = dummy_lr;

    if(my_sub_tree_lr2 == LogicalRegion::NO_REGION)
        my_sub_tree_lr2 = dummy_lr;

    if (my_sub_tree_lr1 == dummy_lr)
        is_left = false;
    if (my_sub_tree_lr2 == dummy_lr)
        is_right = false;

    if (is_left || is_right) {
        assert(my_sub_tree_lr2 != LogicalRegion::NO_REGION);
        assert(my_sub_tree_lr1 != LogicalRegion::NO_REGION);
        assert(my_sub_tree_lr3 != LogicalRegion::NO_REGION);
        assert(lr3 != LogicalRegion::NO_REGION);
        assert(lr2 != LogicalRegion::NO_REGION);
        assert(lr1 != LogicalRegion::NO_REGION);

        GaxpySetTaskArgs args(idx, is_left, is_right);

        TaskLauncher gaxpy_set_task_launcher(GAXPY_SET_TASK_ID, TaskArgument(&args, sizeof(GaxpySetTaskArgs)));

        if (my_sub_tree_lr1 == dummy_lr) {
            RegionRequirement req1(dummy_lr, READ_ONLY, EXCLUSIVE, dummy_lr);
            req1.add_field(FID_X);
            gaxpy_set_task_launcher.add_region_requirement(req1);
        } else {
            RegionRequirement req1(my_sub_tree_lr1, READ_ONLY, EXCLUSIVE, lr1);
            req1.add_field(FID_X);
            gaxpy_set_task_launcher.add_region_requirement(req1);
        }

        if (my_sub_tree_lr2 == dummy_lr) {
            RegionRequirement req2(dummy_lr, READ_ONLY, EXCLUSIVE, dummy_lr);
            req2.add_field(FID_X);
            gaxpy_set_task_launcher.add_region_requirement(req2);
        } else {
            RegionRequirement req2(my_sub_tree_lr2, READ_ONLY, EXCLUSIVE, lr2);
            req2.add_field(FID_X);
            gaxpy_set_task_launcher.add_region_requirement(req2);
        }

        RegionRequirement req3(my_sub_tree_lr3, WRITE_DISCARD, EXCLUSIVE, lr3);
        req3.add_field(FID_X);
        gaxpy_set_task_launcher.add_region_requirement(req3);
        runtime->execute_task(ctx, gaxpy_set_task_launcher);
    }     
}

int norm_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {
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

    lp = runtime->get_logical_partition_by_color(ctx, lr, partition_color);
    LogicalRegion my_sub_tree_lr = runtime->get_logical_subregion_by_color(ctx, lp, my_sub_tree_color);
    LogicalRegion left_sub_tree_lr = runtime->get_logical_subregion_by_color(ctx, lp, left_sub_tree_color);

    IndexSpace indexspace_left = left_sub_tree_lr.get_index_space();
    Future f1;

    if (runtime->has_index_partition(ctx, indexspace_left, partition_color)) {
        idx_left_sub_tree = idx + 1;
        idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));

        assert(lp != LogicalPartition::NO_PART);
        Rect<1> launch_domain(left_sub_tree_color, right_sub_tree_color);
        ArgumentMap arg_map;
        Arguments for_left_sub_tree (n + 1, l * 2    , max_depth, idx_left_sub_tree, partition_color);
        Arguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, partition_color);

        arg_map.set_point(left_sub_tree_color, TaskArgument(&for_left_sub_tree, sizeof(Arguments)));
        arg_map.set_point(right_sub_tree_color, TaskArgument(&for_right_sub_tree, sizeof(Arguments)));

        IndexTaskLauncher norm_launcher(NORM_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        RegionRequirement req(lp, 0, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(FID_X);
        norm_launcher.add_region_requirement(req);
        FutureMap f_result = runtime->execute_index_space(ctx, norm_launcher);
        return f_result.get_result<int>(left_sub_tree_color) + f_result.get_result<int>(right_sub_tree_color);
    } else {
        {
            ReadTaskArgs args(idx);
            TaskLauncher read_task_launcher(READ_TASK_ID, TaskArgument(&args, sizeof(ReadTaskArgs)));
            RegionRequirement req(my_sub_tree_lr, READ_ONLY, EXCLUSIVE, lr);
            req.add_field(FID_X);
            read_task_launcher.add_region_requirement(req);
            f1 = runtime->execute_task(ctx, read_task_launcher);
        }

        return (f1.get_result<int>() * f1.get_result<int>());
    }
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
    LogicalRegion right_sub_tree_lr = runtime->get_logical_subregion_by_color(ctxt, lp, right_sub_tree_color);

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
    IndexSpace indexspace_right = right_sub_tree_lr.get_index_space();



    // These lines will create an instance for the whole region even though we need only the first element
    // const FieldAccessor<READ_ONLY, int, 1> read_acc(regions[0], FID_X);
    // int node_value = read_acc[idx];

    // checking if the children of the node have any valid partition. This condition implies that we are checking if we have reached the leaf node or not
    if (runtime->has_index_partition(ctxt, indexspace_left, partition_color ) 
        || runtime->has_index_partition(ctxt, indexspace_right, partition_color)) { //TODO: probably one side check is enough !

        coord_t idx_left_sub_tree = idx + static_cast<coord_t>(pow(2, n - 1)) + l;
        coord_t idx_right_sub_tree = idx_left_sub_tree + 1;

        // coord_t idx_left_sub_tree = idx + ;
        // coord_t idx_right_sub_tree = idx_left_sub_tree + 1;

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
        TaskVariantRegistrar registrar(UPPER_REFINE_TASK_ID, "upper_refine");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<vector<ReturnRefineTaskArgs>, upper_refine_task>(registrar, "upper_refine");
    }

    {
        TaskVariantRegistrar registrar(SUB_TASKS_REFINE_TASK_ID, "sub_tasks_refine");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<sub_tasks_refine_task>(registrar, "sub_tasks_refine");
    }

    {
        TaskVariantRegistrar registrar(OUTER_REFINE_TASK_ID, "outer_refine");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_inner(true);
        Runtime::preregister_task_variant<outer_refine_task>(registrar, "outer_refine");
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
        TaskVariantRegistrar registrar(DUMMY_COMPRESS_TASK_ID, "dummy_compress");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        // registrar.set_inner(true);
        Runtime::preregister_task_variant<dummy_compress_task>(registrar, "dummy_compress");
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
        TaskVariantRegistrar registrar(GET_COEF_UTIL_TASK_ID, "get_coef_util");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_inner(true);
        Runtime::preregister_task_variant<ReturnGetCoefArguments, get_coef_util_task>(registrar, "get_coef_util");
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

    {
        TaskVariantRegistrar registrar(INNER_PRODUCT_TASK_ID, "inner_product");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_inner(true);
        Runtime::preregister_task_variant<int, inner_product_task>(registrar, "inner_product");
    }

    {
        TaskVariantRegistrar registrar(PRODUCT_TASK_ID, "product");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<int, product_task>(registrar, "product");
    }
    {
        TaskVariantRegistrar registrar(NORM_TASK_ID, "norm");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_inner(true);
        Runtime::preregister_task_variant<int, norm_task>(registrar, "norm");
    }

    {
        TaskVariantRegistrar registrar(GAXPY_SET_TASK_ID, "gaxpy_set");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<gaxpy_set_task>(registrar, "gaxpy_set");
    }

    {
        TaskVariantRegistrar registrar(GAXPY_TASK_ID, "gaxpy");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_inner(true);
        Runtime::preregister_task_variant<gaxpy_task>(registrar, "gaxpy");
    }

    {
        TaskVariantRegistrar registrar(RECONSTRUCT_TASK_ID, "reconstruct");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_inner(true);
        Runtime::preregister_task_variant<reconstruct_task>(registrar, "reconstruct");
    }

    {
        TaskVariantRegistrar registrar(RECONSTRUCT_SET_TASK_ID, "reconstruct_set");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<reconstruct_set_task>(registrar, "reconstruct_set");
    }

    return Runtime::start(argc, argv);
}