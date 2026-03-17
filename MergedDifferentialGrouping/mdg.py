"""
Merged Differential Grouping (MDG) 算法实现 / Implementation of MDG Algorithm.
"""

import numpy as np


class MDG:
    """
    Merged Differential Grouping (MDG) 优化器类 / MDG Optimizer Class.
    """

    def __init__(self, fun, info):
        """
        初始化 MDG 实例 / Initialize MDG instance.
        :param fun: 目标函数对象，需包含 compute 方法 / Objective function object with a compute method.
        :param info: 包含维度(dimension)、上下界(lower/upper)的字典 / Dictionary containing dimension and bounds.
        """
        self.fun = fun
        self.info = info
        self.dim = int(info["dimension"])
        self.lb = info["lower"]
        self.ub = info["upper"]
        self.c = 0.003  # c 是用于控制浮点误差阈值的常数 / c is a constant for floating-point error threshold.
        self.base = (self.ub + self.lb) / 2 - (
            self.ub - self.lb
        ) / 8  # 选择一个基准点 p1 (base)，通常位于搜索空间的中心区域 / Choose a base point p1.
        self.sigma = (
            self.ub - self.lb
        ) / 4  # 定义扰动步长 sigma / Define the perturbation step size sigma.

    def epsilon_calculate(self, fp1, fp2, fp3, fp4):
        """
        计算自适应误差阈值 / Calculate adaptive epsilon threshold.
        利用 IEEE 754 浮点精度 (2^-52) 动态调整，防止数值噪声误判交互作用。
        Uses IEEE 754 precision to prevent numerical noise from misidentifying interactions.
        """
        f_max = max(abs(fp1), abs(fp2), abs(fp3), abs(fp4))
        return self.c * (2**-52) * f_max * self.dim

    def run(self):
        """
        执行 MDG 分解流程 / Execute the MDG decomposition process.
        返回包含可分变量(seps)和不可分变量组(nonseps)的字典。
        Returns a dictionary containing separable variables and non-separable groups.
        """
        dims = []
        seps = []
        all_groups = []
        FEs = 0  # 记录函数评估次数 / Function Evaluations count.

        # 初始化基准点 p1 和全维度扰动点 p4。/ Initialize base point p1 and fully perturbed point p4.
        p1 = np.full(self.dim, self.base)
        fp1 = self.fun.compute(p1)
        p4 = p1 + self.sigma
        fp4 = self.fun.compute(p4)
        FEs += 2

        perturbed_values = np.zeros(self.dim)

        # ---------------------------------------------------------
        # 1. 识别可分变量 / Identify separable variables
        # ---------------------------------------------------------
        for i in range(0, self.dim):
            # p2: 只对第 i 维进行扰动 / Perturb only the i-th dimension.
            p2 = p1.copy()
            p2[i] += self.sigma
            fp2 = self.fun.compute(p2)

            # p3: 在 p4 的基础上撤销第 i 维的扰动 / Reverse i-th perturbation from p4.
            p3 = p4.copy()
            p3[i] -= self.sigma
            fp3 = self.fun.compute(p3)

            FEs += 2

            # 计算增量 / Calculate increments:
            # delta1: 变量 i 在基准环境下的贡献 / Contribution of i at base level.
            # delta2: 变量 i 在其他所有变量均被扰动环境下的贡献 / Contribution of i given other perturbations.
            delta1 = fp2 - fp1
            delta2 = fp4 - fp3

            # 将只扰动变量i的函数值存在perturbed_value中，节省计算资源 / Cache single-dimension perturbation results to reduce FE.
            perturbed_values[i] = fp2[0]

            # 如果变量 i 与其他变量没有交互作用，那么无论其他变量处于什么值，变量 i 对函数值的影响应该是恒定的 / If variable i has no interaction with other variables, then the effect of variable i on the function value should be constant, regardless of the values of the other variables.
            # 若 delta1 == delta2 (在 epsilon 内)，则 i 与其他变量无交互 / If delta1 == delta2, variable i has no interaction with others.
            epsilon = self.epsilon_calculate(fp1, fp2, fp3, fp4)
            if abs(delta1 - delta2) < epsilon:
                seps.append(i + 1)  # 加入可分变量列表 / Add to separable list.
            else:
                dims.append(
                    i + 1
                )  # 加入候选不可分列表 / Add to non-separable candidates

        # ---------------------------------------------------------
        # 2. 递归合并不可分变量 / Recursively merge non-separable variables
        # ---------------------------------------------------------
        if len(dims) > 1:
            # merge_group 会递归地探测 dims 中哪些变量存在真正的相互耦合 / merge_group recursively detects which variables in 'dims' are truly coupled.
            groups, _, FE = self.merge_group(
                self.fun, np.array(dims), fp1, perturbed_values
            )
            FEs += FE

            # 初步判断的不可分变量中可能仍存在可分变量，通过判断分好组的不可分变量中是否有只包含一个变量的组，来避免漏检 /Separable variables may still exist among the initially judged inseparable variables. Missing detection can be avoided by checking whether there exist groups containing only one variable in the grouped inseparable variables.
            all_groups = [g for g in groups if len(g) > 1]
            new_seps = [g[0] for g in groups if len(g) == 1]
            seps.extend(new_seps)

        subspaces = {"nonseps": all_groups, "seps": seps}
        return subspaces

    @staticmethod
    def merge_interaction_group(LgroupIndexs, Rinteract_Indexs, Lgroups, Rgroups):
        """
        根据交互索引合并左右变量组 / Merge L/R groups based on interaction indices.
        这是一个基于传递性的合并过程（如果 A 与 B 交互，B 与 C 交互，则 {A, B, C} 为一组）。
        A transitive closure merge (If A-B interact and B-C interact, then {A,B,C} is one group).
        """
        merged_groups = []
        LgroupIndexs = list(LgroupIndexs)
        Rinteract_Indexs = list(Rinteract_Indexs)

        while LgroupIndexs:
            cur_index = LgroupIndexs[0]
            # 获取当前左组对应的所有与其有交互的右组索引 / Get indices of all Right groups interacting with the current Left group.
            cur_interact_indexs = list(Rinteract_Indexs[0])
            Rightgroup_index = list(Rinteract_Indexs[0])

            left_item = Lgroups[cur_index]
            cur_Leftgroup = (
                left_item.tolist()
                if isinstance(left_item, np.ndarray)
                else list(left_item)
            )

            del LgroupIndexs[0]
            del Rinteract_Indexs[0]

            # 查找并合并所有有相关变量的左组 / Identify and merge all left groups that contain relevant variables.
            i = 0
            while i < len(LgroupIndexs):
                if set(cur_interact_indexs) & set(Rinteract_Indexs[i]):
                    next_left = Lgroups[LgroupIndexs[i]]
                    next_left_list = (
                        next_left.tolist()
                        if isinstance(next_left, np.ndarray)
                        else list(next_left)
                    )
                    cur_Leftgroup.extend(next_left_list)

                    # 并集更新右组索引 / Union update Right group indices.
                    Rightgroup_index = list(
                        set(Rightgroup_index) | set(Rinteract_Indexs[i])
                    )
                    del LgroupIndexs[i]
                    del Rinteract_Indexs[i]
                else:
                    i += 1
            # 加入所有相关的右组 / Add all related right groups
            cur_Rightgroup = []
            for j in Rightgroup_index:
                right_item = Rgroups[j]
                right_item_list = (
                    right_item.tolist()
                    if isinstance(right_item, np.ndarray)
                    else list(right_item)
                )
                cur_Rightgroup.extend(right_item_list)

            merged_groups.append(cur_Leftgroup + cur_Rightgroup)

        return merged_groups

    def bisearch(self, fun, left_group, fp1, fp2, fp_iR, Rgroups, Rperts_cache):
        """
        二分搜索识别交互组 / Binary search to identify interacting groups.
        用于高效查找 left_group 具体与 Rgroups 中的哪些子组存在交互。
        Efficiently finds which specific subgroups in Rgroups interact with 'left_group'.
        """
        FEs = 0
        R_gnum = len(Rgroups)
        Rperts = [0] * (4 * self.dim)  # 函数值缓存 / Function value cache.
        Rperts[0] = Rperts_cache
        Rfp = Rperts[0]

        data = [
            fp1,
            fp2,
            Rfp,
            fp_iR,
        ]  # 函数值队列 [f1, f2, f3, f4] 用于交互检测。 / Function value queue [f1​,f2​,f3​,f4​] used for interaction detection.
        groupsQueue = [Rgroups]
        dataQueue = [data.copy()]
        groupIndexsQueue = [list(range(0, R_gnum))]
        pQueue = [np.full(self.dim, self.base)]
        node_orders = [1]
        group = []
        groupIndexs = []

        # 递归二分搜索 / Recursive binary search
        while groupsQueue:
            cur_groups = groupsQueue.pop(0)
            cur_data = dataQueue.pop(0)
            cur_groupIndexs = groupIndexsQueue.pop(0)
            cur_p = pQueue.pop(0)
            cur_order = node_orders.pop(0)

            delta1 = cur_data[1] - cur_data[0]
            delta2 = cur_data[3] - cur_data[2]
            epsilon = self.epsilon_calculate(
                cur_data[0], cur_data[1], cur_data[2], cur_data[3]
            )

            # 如果检测到交互作用 / If interaction is detected:
            if abs(delta1 - delta2) > epsilon:
                cur_gnum = len(cur_groups)
                if cur_gnum == 1:
                    # 递归到底部，确定了最小交互组 / Reached leaf, found interacting group.
                    group.append(cur_groups[0])
                    groupIndexs.append(cur_groupIndexs[0])
                else:
                    # 继续二分拆分 / Continue bisection.
                    median = cur_gnum // 2
                    groups1 = cur_groups[0:median]
                    groupIndexs1 = cur_groupIndexs[0:median]

                    # 准备左子树的探测点 / Prepare probe points for left child.
                    p_1 = cur_p.copy()
                    p_1[[i - 1 for g in groups1 for i in g]] = self.base + self.sigma

                    # 读取缓存或计算函数值 / Check cache or compute.
                    if cur_order * 2 <= len(Rperts) and Rperts[cur_order * 2 - 1] != 0:
                        fp_1 = Rperts[cur_order * 2 - 1]
                    else:
                        fp_1 = fun.compute(p_1)
                        Rperts[cur_order * 2 - 1] = fp_1
                        FEs += 1

                    p_i1 = p_1.copy()
                    p_i1[np.array(left_group) - 1] = self.base + self.sigma
                    fp_i1 = fun.compute(p_i1)
                    FEs += 1

                    # 将左右子区间入列/ Enqueue children.
                    data1 = [cur_data[0], cur_data[1], fp_1, fp_i1]
                    groupsQueue.append(groups1)
                    dataQueue.append(data1)
                    groupIndexsQueue.append(groupIndexs1)
                    pQueue.append(cur_p)
                    node_orders.append(cur_order * 2)

                    groups2 = cur_groups[median:cur_gnum]
                    groupIndexs2 = cur_groupIndexs[median:cur_gnum]
                    data2 = [fp_1, fp_i1, cur_data[2], cur_data[3]]
                    groupsQueue.append(groups2)
                    dataQueue.append(data2)
                    groupIndexsQueue.append(groupIndexs2)
                    pQueue.append(p_1)
                    node_orders.append(cur_order * 2 + 1)
        return group, groupIndexs, Rperts, FEs

    def merge_group(self, fun, dims, fp1, perturbed_values):
        """
        递归拆分并合并变量组 / Recursively split and merge variable groups.
        这是 MDG 的核心递归逻辑。
        The core recursive logic of MDG.
        """
        FEs = 0
        dim_len = len(dims)
        groups = []

        # 基准情况：1或2个变量 / Base case: 1 or 2 variables
        if dim_len == 1 or dim_len == 2:
            if dim_len == 1:
                groups_perturb = perturbed_values[int(dims[0]) - 1].copy()
                groups.append(dims.tolist())
            else:
                p = np.full(self.dim, self.base)
                p[dims - 1] = self.base + self.sigma
                fp = fun.compute(p)
                FEs += 1

                # 检查两个变量之间是否存在直接交互 / Check for direct interaction between the two variables.
                delta1 = perturbed_values[dims[0] - 1] - fp1
                delta2 = fp - perturbed_values[dims[1] - 1]
                epsilon = self.epsilon_calculate(
                    fp1,
                    perturbed_values[dims[0] - 1],
                    perturbed_values[dims[1] - 1],
                    fp,
                )

                groups_perturb = fp
                if abs(delta1 - delta2) < epsilon:
                    groups.append([dims[0]])
                    groups.append([dims[1]])
                else:
                    groups = [dims.tolist()]
            return groups, groups_perturb, FEs

        # 递归拆分：将当前维度集对半拆分为 L 和 R / Recursive split: split current dims into L and R halves.
        median = dim_len // 2
        Ldims = dims[0:median]
        Rdims = dims[median:dim_len]

        # 递归获取左右两侧的子组划分 / Recursively get subgroup partitions for Left and Right sides.
        Lgroups, Lgroups_perturb, LFEs = self.merge_group(
            fun, Ldims, fp1, perturbed_values
        )
        Rgroups, Rgroups_perturb, RFEs = self.merge_group(
            fun, Rdims, fp1, perturbed_values
        )
        FEs = LFEs + RFEs

        L_gnum = len(Lgroups)
        R_gnum = len(Rgroups)

        Lfp = Lgroups_perturb
        Rfp = Rgroups_perturb

        # 检查 L 集合与 R 集合整体是否存在交互 / Check if the set L and set R interact as a whole.
        p = np.full(self.dim, self.base)
        p[dims - 1] = self.base + self.sigma
        fp = fun.compute(p)
        FEs += 1

        delta1 = Lfp - fp1
        delta2 = fp - Rfp
        epsilon = self.epsilon_calculate(fp1, Lfp, Rfp, fp)
        groups_perturb = fp

        if abs(delta1 - delta2) > epsilon:
            # 存在交互，调用二分查找 / Interaction exists, call bisearch
            if L_gnum == 1 and R_gnum == 1:
                # 简单情况：两边都只有一个组，直接合并 / Simple case: both sides have one group, merge directly.
                Rgroups[0] = Lgroups[0] + Rgroups[0]
                groups = Rgroups
                return groups, groups_perturb, FEs

            # 找出所有与右侧有交互的左侧组
            Lgroup, LgroupIndexs, _, FE = self.bisearch(
                fun, Rdims, fp1, Rfp, fp, Lgroups, Lfp
            )
            FEs += FE
            if LgroupIndexs:
                lnum = len(LgroupIndexs)
                if R_gnum == 1:
                    # 如果右侧只有一个组，所有检测到的左组都并入其中 / If Right side is a single group, merge all detected Left groups into it.
                    for group in Lgroup:
                        for i in group:
                            Rgroups[0].append(i)
                    LgroupIndexs.sort(reverse=True)
                    for i in LgroupIndexs:
                        del Lgroups[i]
                    groups = Lgroups + Rgroups
                    return groups, groups_perturb, FEs

                # 对每一个有交互的左组，进一步通过二分查找确定它关联了哪些右组 / For each interacting Left group, find its associated Right groups through bisearch.
                Rinteract_Indexs = [[] for _ in range(lnum)]
                Rperts_cache = Rfp
                for i in range(lnum):
                    # 计算当前左组的扰动值用于二分查找 / Calculate the perturbation value of the current left group for bisearch.
                    if L_gnum == 1:
                        fp_i = Lfp
                        fp_iR = fp
                    else:
                        p_i = np.full(self.dim, self.base)
                        p_i[np.array(Lgroup[i]) - 1] = self.base + self.sigma
                        fp_i = fun.compute(p_i)
                        p_iR = p_i.copy()
                        p_iR[[r - 1 for r in Rdims]] = self.base + self.sigma
                        fp_iR = fun.compute(p_iR)
                        FEs += 2

                    _, RgroupIndexs, _, FE = self.bisearch(
                        fun,
                        Lgroup[i],
                        fp1,
                        fp_i,
                        fp_iR,
                        Rgroups,
                        Rperts_cache,
                    )
                    FEs += FE
                    Rinteract_Indexs[i] = RgroupIndexs

                # 执行最终合并 / Perform the final merge of groups with identified interactions.
                LgroupIndexs_cache = LgroupIndexs.copy()
                Rinteract_Indexs_cache = Rinteract_Indexs.copy()

                merged_groups = self.merge_interaction_group(
                    LgroupIndexs_cache, Rinteract_Indexs_cache, Lgroups, Rgroups
                )

                # 移除已合并的旧组并更新 / Remove merged old groups and update
                LgroupIndexs.sort(reverse=True)
                for j in LgroupIndexs:
                    del Lgroups[j]
                list_r = []
                for k in range(lnum):
                    list_r = list(set(list_r) | set(Rinteract_Indexs[k]))
                list_r.sort(reverse=True)
                for j in list_r:
                    del Rgroups[j]
                Rgroups = Rgroups + merged_groups

        # 返回当前递归层级的最终分组结果 / Return the final grouping result for the current recursion level.
        groups = Lgroups + Rgroups
        return groups, groups_perturb, FEs
