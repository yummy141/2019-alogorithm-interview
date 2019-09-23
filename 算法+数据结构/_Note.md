## 归并排序
- 递推公式
  - `merge(A, p, q) = merge(mergesort(A, p, r) + mergesort(A, r+1, q))`
  - 终止条件：`p >= q` 
- 归并排序是稳定的算法
- 最好、最坏、平均时间复杂度为O(nlogn)
- 空间复杂度为O(n)

## 快速排序
- 递推公式
  - `qsort(A, l, r) = qsort(A, l, p-1) + p + qsort(A, p+1, r) `
  - 终止条件：`l >= r`
- 快排是不稳定的算法
- 时间复杂度一般为O(nlogn)
- 快排是原地排序算法，空间复杂度为O(1)
> CSDN/[快速排序(三种算法实现和非递归实现)](https://blog.csdn.net/qq_36528114/article/details/78667034)

## 排序复杂度和稳定性分析

<div align="center"><img src="../_image/sort_fig1.png" width=""/></div>

**典型算法题**
Q1： 如何用快排思想在O(n)内查找第K大元素？
Q2： 现在有10个接口访问日志文件，每个日志文件大小约300MB，每个文件里的日志都是按照时间戳从小到大排序的，你希望将这10个较小的日志文件，合并为1个日志文件，合并之后的日志仍然按照时间戳从小到大排列。内存只有1GB
A2： 桶排序
Q3: 对 D，a，F，B，c，A，z 这个字符串排序，小写要在大写前面，内部不要求排序顺序？
A3: 双指针


## 堆的应用
```c++
void adjust(vector<int>& nums, const int n, int index){
    // int n = nums.size();
    if(index >= n)
        return;
    int l = 2 * index + 1;
    int r = 2 * index + 2;
    int max_index = index;
    if(l < n && nums[max_index] < nums[l])
        max_index = l;
    if(r < n && nums[max_index] < nums[r])
        max_index = r;

    if(max_index != index){
        swap(nums[index], nums[max_index]);
        adjust(nums, n, max_index);
    }
}

void heapSort(vector<int>& nums){
    int n = nums.size();
    // build
    for(int i = n  / 2 - 1; i >= 0; i--)
        adjust(nums, n, i);

    for(int i = n - 1; i > 0; i--){
        swap(nums[0], nums[i]);
        adjust(nums, i, 0);
    }
}

int main() {
    vector<int> t = {9, 1, 5, 7, 3, 2};
    heapSort(t);
    for(auto i : t)
        cout << i << " ";
}
```

- 建堆时间复杂度O(N)
- 堆删除和插入的时间复杂度都是O(logN)
- 可以用来合并有序小文件
  - 每次从每个小文件中取出一个数据，比较出最小数据，再放入大文件中
- 高性能定时器
  - 不用每隔1s轮询任务列表
- Top K
- 求中位数

## 快速排序
```c++
int partition(vector<int>& nums, int l, int r){
    int index = rand() % (r - l + 1) + l;
    int pivot = nums[index];
    swap(nums[r], nums[index]);
    int i = l;
    for(int j = l; j < r; j++){
        if(nums[j] < pivot)
            swap(nums[i++], nums[j]);
    }
    swap(nums[r], nums[i]);
    return i;
}

void quicksort(vector<int>& nums, int l, int r){
    if(l >= r)
        return;

    int k = partition(nums, l, r);
    if(k > l)
        quicksort(nums, l, k - 1);
    if(k < r)
        quicksort(nums, k + 1, r);

}

int main() {
    vector<int> t = {9, 1, 5, 7, 3, 2};
    int n = t.size();
    quicksort(t, 0, n - 1);
    for(auto i : t)
        cout << i << " ";
}
```