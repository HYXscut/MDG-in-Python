import numpy as np


def epsilonCalculate(fp1, fp2, fp3, fp4, dim):
    Fmax = max(abs(fp1), abs(fp2), abs(fp3), abs(fp4))
    c = 0.003
    epsilon = c * (2**-52) * Fmax * dim
    return epsilon


class MDG:
    def __init__(self, fun, info):
        self.fun = fun
        self.info = info

    def run(self):
        lb = self.info["lower"]
        ub = self.info["upper"]
        dim = self.info["dimension"]
        base = (ub + lb) / 2 - (ub - lb) / 8
        sigma = (ub - lb) / 4

        dims = []
        seps = []
        allgroups = []
        FEs = 0
        perturbed_values = []

        p1 = np.full(dim, base)
        fp1 = self.fun.compute(p1)
        p4 = p1 + sigma
        fp4 = self.fun.compute(p4)

        FEs += 2
        perturbed_values = np.zeros(dim)

        for i in range(0, dim):
            p2 = p1.copy()
            p2[i] += sigma
            fp2 = self.fun.compute(p2)

            p3 = p4.copy()
            p3[i] -= sigma
            fp3 = self.fun.compute(p3)

            FEs += 2

            delta1 = fp2 - fp1
            delta2 = fp4 - fp3
            # 将只扰动变量i的函数值存在perturbed_value中，提高运行效率
            perturbed_values[i] = fp2[0]
            # 如果变量 i 与其他变量没有交互作用，那么无论其他变量处于什么值，变量 i 对函数值的影响应该是恒定的。
            epsilon = epsilonCalculate(fp1, fp2, fp3, fp4, dim)
            if abs(delta1 - delta2) < epsilon:
                seps.append(i + 1)
            else:
                dims.append(i + 1)

        if len(dims) > 1:
            groups, _, FE = mergeGroup(
                self.fun, self.info, np.array(dims), fp1, perturbed_values
            )
            FEs += FE
            # 初步判断的不可分变量中可能仍存在可分变量，通过判断分好组的不可分变量中是否有只包含一个变量的组，来避免漏检

            allgroups = [g for g in groups if len(g) > 1]
            new_seps = [g[0] for g in groups if len(g) == 1]
            seps.extend(new_seps)

        subspaces = {"nonseps": allgroups, "seps": seps}
        return subspaces


def mergeInteractionGroup(LgroupIndexs, Rinteract_Indexs, Lgroups, Rgroups):
    merged_groups = []
    cur_Leftgroup = []
    LgroupIndexs = list(LgroupIndexs)
    Rinteract_Indexs = list(Rinteract_Indexs)

    while LgroupIndexs:
        cur_Rightgroup = []
        cur_index = LgroupIndexs[0]
        cur_interact_indexs = list(Rinteract_Indexs[0])
        Rightgroup_index = list(Rinteract_Indexs[0])

        left_item = Lgroups[cur_index]
        cur_Leftgroup = left_item.tolist() if isinstance(left_item, np.ndarray) else list(left_item)

        del LgroupIndexs[0]
        del Rinteract_Indexs[0]
        
        i = 0
        while i < len(LgroupIndexs):
            if set(cur_interact_indexs) & set(Rinteract_Indexs[i]):
                next_left = Lgroups[LgroupIndexs[i]]
                next_left_list = next_left.tolist() if isinstance(next_left, np.ndarray) else list(next_left)
                cur_Leftgroup.extend(next_left_list)

                Rightgroup_index = list(set(Rightgroup_index) | set(Rinteract_Indexs[i]))
                del LgroupIndexs[i]
                del Rinteract_Indexs[i]
            else:
                i += 1
        
        for j in Rightgroup_index:
            right_item = Rgroups[j]
            right_item_list = right_item.tolist() if isinstance(right_item, np.ndarray) else list(right_item)
            cur_Rightgroup.extend(right_item_list)
        
        merged_groups.append(cur_Leftgroup + cur_Rightgroup)
        
    return merged_groups


def bisearch(fun, info, left_group, fp1, fp2, fp_iR, Rgroups, Rperts_cache):
    lb = info["lower"]
    ub = info["upper"]
    dim = info["dimension"]
    base = (ub + lb) / 2 - (ub - lb) / 8
    sigma = (ub - lb) / 4
    FEs = 0
    R_gnum = len(Rgroups)
    Rperts = [0] * (4 * dim)
    Rperts[0] = Rperts_cache
    Rfp = Rperts[0]

    data = [fp1, fp2, Rfp, fp_iR]
    groupsQueue = [Rgroups]
    dataQueue = [data.copy()]
    groupIndexsQueue = [list(range(0, R_gnum))]
    pQueue = [np.full(dim, base)]
    node_orders = [1]
    group = []
    groupIndexs = []

    while groupsQueue:
        cur_groups = groupsQueue.pop(0)
        cur_data = dataQueue.pop(0)
        cur_groupIndexs = groupIndexsQueue.pop(0)
        cur_p = pQueue.pop(0)
        cur_order = node_orders.pop(0)

        delta1 = cur_data[1] - cur_data[0]
        delta2 = cur_data[3] - cur_data[2]
        epsilon = epsilonCalculate(
            cur_data[0], cur_data[1], cur_data[2], cur_data[3], dim
        )

        if abs(delta1 - delta2) > epsilon:
            cur_gnum = len(cur_groups)
            if cur_gnum == 1:
                group.append(cur_groups[0])
                groupIndexs.append(cur_groupIndexs[0])
            else:
                median = cur_gnum // 2
                groups1 = cur_groups[0:median]
                groupIndexs1 = cur_groupIndexs[0:median]
                p_1 = cur_p.copy()

                p_1[[i - 1 for g in groups1 for i in g]] = base + sigma

                if cur_order * 2 <= len(Rperts) and Rperts[cur_order * 2 - 1] != 0:
                    fp_1 = Rperts[cur_order * 2 - 1]
                else:
                    fp_1 = fun.compute(p_1)
                    Rperts[cur_order * 2 - 1] = fp_1
                    FEs += 1
                p_i1 = p_1.copy()
                p_i1[np.array(left_group) - 1] = base + sigma
                fp_i1 = fun.compute(p_i1)
                FEs += 1

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


def mergeGroup(fun, info, dims, fp1, perturbed_values):
    lb = info["lower"]
    ub = info["upper"]
    dim = info["dimension"]
    base = (ub + lb) / 2 - (ub - lb) / 8
    sigma = (ub - lb) / 4
    FEs = 0
    dim_len = len(dims)
    groups = []

    if dim_len == 1 or dim_len == 2:
        if dim_len == 1:
            groups_perturb = perturbed_values[int(dims[0]) - 1].copy()
            groups.append(dims.tolist())
        else:
            p = np.full(dim, base)
            p[dims - 1] = base + sigma
            fp = fun.compute(p)
            FEs += 1

            # 变量2不扰动，只扰动变量1产生的变化
            delta1 = perturbed_values[dims[0] - 1] - fp1
            # 变量2已扰动，再扰动变量1产生的变化
            delta2 = fp - perturbed_values[dims[1] - 1]
            epsilon = epsilonCalculate(
                fp1,
                perturbed_values[dims[0] - 1],
                perturbed_values[dims[1] - 1],
                fp,
                dim,
            )

            groups_perturb = fp
            if abs(delta1 - delta2) < epsilon:
                groups.append([dims[0]])
                groups.append([dims[1]])
            else:
                groups = [dims.tolist()]
        return groups, groups_perturb, FEs

    median = dim_len // 2
    Ldims = dims[0:median]
    Rdims = dims[median:dim_len]

    Lgroups, Lgroups_perturb, LFEs = mergeGroup(fun, info, Ldims, fp1, perturbed_values)
    Rgroups, Rgroups_perturb, RFEs = mergeGroup(fun, info, Rdims, fp1, perturbed_values)

    FEs = LFEs + RFEs
    L_gnum = len(Lgroups)
    R_gnum = len(Rgroups)

    Lfp = Lgroups_perturb
    Rfp = Rgroups_perturb

    p = base * np.ones(dim)
    p[dims - 1] = base + sigma
    fp = fun.compute(p)
    FEs += 1

    delta1 = Lfp - fp1
    delta2 = fp - Rfp
    epsilon = epsilonCalculate(fp1, Lfp, Rfp, fp, dim)
    groups_perturb = fp

    if abs(delta1 - delta2) > epsilon:
        if L_gnum == 1 and R_gnum == 1:
            Rgroups[0] = Lgroups[0] + Rgroups[0]
            groups = Rgroups
            return groups, groups_perturb, FEs

        # 逆用bisearch逻辑，先找出所有与右侧有交互的左侧组
        Lgroup, LgroupIndexs, _, FE = bisearch(
            fun, info, Rdims, fp1, Rfp, fp, Lgroups, Lfp
        )
        FEs += FE
        if LgroupIndexs:
            lnum = len(LgroupIndexs)
            if R_gnum == 1:
                for group in Lgroup:
                    for i in group:
                        Rgroups[0].append(i)
                LgroupIndexs.sort(reverse=True)
                for i in LgroupIndexs:
                    del Lgroups[i]
                groups = Lgroups + Rgroups
                return groups, groups_perturb, FEs
            Rinteract_Indexs = [[] for _ in range(lnum)]
            Rperts_cache = Rfp
            for i in range(lnum):
                if L_gnum == 1:
                    fp_i = Lfp
                    fp_iR = fp
                else:
                    p_i = base * np.ones(dim)
                    p_i[np.array(Lgroup[i]) - 1] = base + sigma
                    fp_i = fun.compute(p_i)
                    p_iR = p_i.copy()
                    # iR=Lgroup[i]+Rdims(改进：只需给右组加扰动)
                    p_iR[[r - 1 for r in Rdims]] = base + sigma
                    fp_iR = fun.compute(p_iR)
                    FEs += 2

                _, RgroupIndexs, Rperts, FE = bisearch(
                    fun,
                    info,
                    Lgroup[i],
                    fp1,
                    fp_i,
                    fp_iR,
                    Rgroups,
                    Rperts_cache,
                )
                FEs += FE
                Rinteract_Indexs[i] = RgroupIndexs

            LgroupIndexs_cache = LgroupIndexs.copy()
            Rinteract_Indexs_cache = Rinteract_Indexs.copy()
            merged_groups = mergeInteractionGroup(
                LgroupIndexs_cache, Rinteract_Indexs_cache, Lgroups, Rgroups
            )
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
    groups = Lgroups + Rgroups
    return groups, groups_perturb, FEs
