目录
---
<!-- TOC -->

- [数组中重复的数字](#数组中重复的数字)
- [二维数组中的查找](#二维数组中的查找)
- [从尾到头打印链表](#从尾到头打印链表)
- [重建二叉树](#重建二叉树)
- [二叉树的下一个节点](#二叉树的下一个节点)
- [用两个栈实现队列](#用两个栈实现队列)
- [跳台阶](#跳台阶)
- [跳台阶II](#跳台阶ii)
- [矩形覆盖](#矩形覆盖)
- [旋转数组的最小数字](#旋转数组的最小数字)
- [矩阵中的路径](#矩阵中的路径)
- [整数拆分(动态规划|贪心)](#整数拆分动态规划贪心)
- [二进制中1的个数（位运算）](#二进制中1的个数位运算)
- [数值的整数次方（位运算）](#数值的整数次方位运算)
- [删除链表中重复的节点](#删除链表中重复的节点)
- [调整数组顺序使奇数位于偶数前面（数组）](#调整数组顺序使奇数位于偶数前面数组)
- [链表中倒数第k个结点](#链表中倒数第k个结点)
- [链表中环的入口节点（链表）](#链表中环的入口节点链表)
- [反转链表](#反转链表)
- [合并两个排序的链表](#合并两个排序的链表)
- [二叉树的镜像](#二叉树的镜像)
- [对称的二叉树](#对称的二叉树)
- [顺时针打印矩阵](#顺时针打印矩阵)
- [包含min函数的栈（数据结构）](#包含min函数的栈数据结构)
- [栈的压入、弹出序列](#栈的压入弹出序列)
- [从上往下打印二叉树](#从上往下打印二叉树)
- [把二叉树打印成多行](#把二叉树打印成多行)
- [按之字形顺序打印二叉树](#按之字形顺序打印二叉树)
- [二叉搜索树的后序遍历序列](#二叉搜索树的后序遍历序列)
- [二叉搜索树的第k个结点](#二叉搜索树的第k个结点)
- [二叉树中和为某一值的路径](#二叉树中和为某一值的路径)
- [复杂链表的控制](#复杂链表的控制)
- [平衡二叉树](#平衡二叉树)
- [二叉搜索树与双向链表](#二叉搜索树与双向链表)
- [序列化二叉树](#序列化二叉树)
- [字符串的排列](#字符串的排列)
- [数组中超过一半的数字](#数组中超过一半的数字)
- [数组中第k大的元素](#数组中第k大的元素)
- [数据流中的中位数](#数据流中的中位数)
- [数字1的个数](#数字1的个数)
- [连续子数组的最大和](#连续子数组的最大和)
- [把数组排成最小的数](#把数组排成最小的数)
- [把数字翻译成字符串（解码方法）](#把数字翻译成字符串解码方法)
- [礼物的最大价值](#礼物的最大价值)
- [丑数](#丑数)
- [丑数II](#丑数ii)
- [第一个只出现一次的字符](#第一个只出现一次的字符)
- [字符流中第一个只出现一次的字符](#字符流中第一个只出现一次的字符)
- [替换空格](#替换空格)
- [翻转单词序列](#翻转单词序列)
- [不用加减乘除做加法](#不用加减乘除做加法)
- [扑克牌顺子](#扑克牌顺子)

<!-- /TOC -->
## 数组中重复的数字
[**题目描述**](https://www.nowcoder.com/practice/623a5ac0ea5b4e5f95552655361ae0a8?tpId=13&tqId=11203&tPage=3&rp=3&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
```
在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。
```

**思路**
- 时间复杂度O(N), 空间复杂度O(1)
    - 因此不能使用诸如map,set等
- 正确的思路即为以数组下标当成顺序标记，遍历的同时将数据swap到正确的下标下，直到swap时发现当前下标和目标下标里两个数据已经相等，则返回true
```c++
class Solution {
public:
    // Parameters:
    //        numbers:     an array of integers
    //        length:      the length of array numbers
    //        duplication: (Output) the duplicated number in the array number
    // Return value:       true if the input is valid, and there are some duplications in the array number
    //                     otherwise false
    bool duplicate(int numbers[], int length, int* duplication) {
        if(length <= 0) 
            return false;
        
        for(int i = 0; i < length; i++){
            while(numbers[i]!=i){
                if(numbers[i]==numbers[numbers[i]]){
                    *duplication = numbers[i];
                    return true;
                }
                swap(numbers[i], numbers[numbers[i]]);
            }
        }
        return false;
    }
};
```

## 二维数组中的查找

**描述**
```
在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
```
```c++
class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        int n = array.size();
        int m = array[0].size();
        
        int i = 0;
        int j = m - 1;
        while((i < n) && (j >= 0)){
            if(array[i][j] == target)
                return true;
            else if(array[i][j] > target)
                j--;
            else
                i++;
        }
        return false;
    }
};
```

## 从尾到头打印链表
[链接](https://www.nowcoder.com/practice/d0267f7f55b3412ba93bd35cfa8e8035?tpId=13&tqId=11156&tPage=1&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)

**思路**
- 栈
    - 或者直接用动态数组插入，然后返回(头插法)
```c++
/**
*  struct ListNode {
*        int val;
*        struct ListNode *next;
*        ListNode(int x) :
*              val(x), next(NULL) {
*        }
*  };
*/
class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        vector<int> ret;
        ListNode* p = head;
        while(p!=NULL){
            ret.insert(ret.begin(), p->val);
            p = p->next;
        }
        return ret;
    }
};
```

## 重建二叉树
>[Nowcoder](https://www.nowcoder.com/practice/8a19cbe657394eeaac2f6ea9b0f6fcf6?tpId=13&tqId=11157&tPage=1&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)
**示例**
```
前序
  1,2,4,7,3,5,6,8
中序
  4,7,2,1,5,3,8,6

第一层
  根节点 1
  根据根节点的值（不重复），划分中序：
    {4,7,2} 和 {5,3,8,6}
  根据左右子树的长度，划分前序：
    {2,4,7} 和 {3,5,6,8}
  从而得到左右子树的前序和中序
    左子树的前序和中序：{2,4,7}、{4,7,2}
    右子树的前序和中序：{3,5,6,8}、{5,3,8,6}

第二层
  左子树的根节点 2
  右子树的根节点 3
  ...
```

- 这个方法比较浪费空间，传数组引用会更好
```c++
/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
        if(pre.size() <= 0)
            return NULL;
        
        TreeNode* t = new TreeNode{ pre[0] };
        for(int i = 0; i < pre.size(); i++){
            if(vin[i] == pre[0]){
                t->left = reConstructBinaryTree(vector<int>(pre.begin() + 1, pre.begin() + 1 + i), vector<int>(vin.begin(), vin.begin() + i));
                t->right = reConstructBinaryTree(vector<int>(pre.begin() + 1 + i, pre.end()), vector<int>(vin.begin() + 1 + i, vin.end()));
            }
        }
        return t;
    }
};
```

## 二叉树的下一个节点
> [Nowcoder](https://www.nowcoder.com/practice/9023a0c988684a53960365b889ceaf5e?tpId=13&tqId=11210&tPage=3&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)

**描述**  
```
给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
```

**思路**
- 如果一个节点的右子树不为空，那么下一个节点是该节点右子树的最左叶子；
- 否则（右子树为空），沿父节点向上直到找到某个节点是其父节点的左孩子，那么该父节点就是下一个节点

**解释**
- 当前节点为遍历到的节点，如果它是叶节点（既没有左子树也没有右子树）
    - 且如果是父节点的左节点
        - 则其父节点为下一个遍历节点
    - 且如果是父节点的右节点
        - 则递归找到第一个是有左节点的节点
- 如果不是叶节点
    - 根据左根右原则，当前节点左子树部分（如果有）及自身已经遍历完，则考虑其右子树
        - 如果有右子树，则按照左根右原则去找右子树中的最左节点
        - 如果没有右子树
            - 则可当成叶节点处理
```c++
/*
struct TreeLinkNode {
    int val;
    struct TreeLinkNode *left;
    struct TreeLinkNode *right;
    struct TreeLinkNode *next;
    TreeLinkNode(int x) :val(x), left(NULL), right(NULL), next(NULL) {
        
    }
};
*/
class Solution {
public:
    TreeLinkNode* GetNext(TreeLinkNode* pNode){
        if(pNode == NULL)
            return NULL;
        
        if(pNode->right != NULL){
            auto p = pNode->right;
            while(p->left){
                p = p->left;
            }
            return p;
        }
        else{
            auto p = pNode;
            while(p->next != NULL){
                if(p->next->left == p)
                    return p->next;
                p = p->next;
            }
        }
        return NULL;
    }
};
```

## 用两个栈实现队列
> [Nowcoder](https://www.nowcoder.com/practice/54275ddae22f475981afa2244dd448c6?tpId=13&tqId=11158&tPage=1&rp=3&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)

**描述**
```
用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
```

**解释**
- 注意出队列，要把输出栈中的元素全部输出后才到第一个栈中找元素
```c++
class Solution
{
public:
    void push(int node) {
        stack1.push(node);
    }

    int pop() {
        if(stack2.size() == 0){
            while(stack1.size() > 0){
                auto t = stack1.top();
                stack1.pop();
                stack2.push(t);
            }
        }
        auto t = stack2.top();
        stack2.pop();
        return t;
    }

private:
    stack<int> stack1;
    stack<int> stack2;
};
```

## 跳台阶
>[Nowcoder](https://www.nowcoder.com/practice/8c82a5b80378478f9484d87d1c5f12a4?tpId=13&tqId=11161&tPage=1&rp=3&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)

**描述**
```
一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
```
```c++
class Solution {
public:
    int jumpFloor(int number) {
        //斐波那契数列
        int f1 = 1;
        int f2 = 2;
        number--;
        while(number--){
            f2 = f1 + f2;
            f1 = f2 - f1;
        }
        return f1;
    }
};
```

## 跳台阶II

**描述**
```
一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
```
```c++
class Solution {
public:
    int jumpFloorII(int number) {
        vector<int> dp(number+1, 1);
        for(int i = 1; i <= number; i++)
            for(int j = 1; j < i ; j++)
                dp[i] += dp[j];
        return dp[number];
    }
};
```

## 矩形覆盖
> NowCoder/[矩形覆盖](https://www.nowcoder.com/practice/72a5a919508a4251859fb2cfb987a0e6?tpId=13&tqId=11163&tPage=1&rp=3&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)

**描述**
```
我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
```

```c++
class Solution {
public:
    int rectCover(int number) {
        if(number <= 0) 
            return 0;
        if(number == 1) 
            return 1;
        int f = 1;
        int g = 2;
        for(int i = 3; i <= number; i++){
            g = g + f;
            f = g - f;
        }
        return g;
    }
};
```

## 旋转数组的最小数字
> NowCoder/[旋转数组的最小数字](https://www.nowcoder.com/practice/9f3231a991af4f55b95579b44b7a01ba?tpId=13&tqId=11159&tPage=1&rp=3&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)

**描述**
```
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
```
- 二分查找：注意有重复的情况
```c++
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        int r =  rotateArray.size() - 1;
        if (r == -1) return 0;
        int l = 0;
        while(l <= r){
            int mid = l + (r - l) / 2;
            if(rotateArray[l] < rotateArray[r]){
                return rotateArray[l];
            }
            // 只有两位的时候需要特殊考虑
            if((r - l) == 1){ 
                return rotateArray[r];
            }
            
            // 如果中间位和左边和右边都相等，则无法判断最小值在哪一侧，只能直接遍历
            if((rotateArray[mid] == rotateArray[l]) && (rotateArray[mid] == rotateArray[r])) {
                int ret = INT_MAX;
                for(int i = l; i <= r; i++){
                    ret = min(ret, rotateArray[i]);
                }
                return ret;
            }
            
            // 注意这里一定要加等号，假设Array[mid] == Array[r], 则 Array[l]一定大于Array[mid]
            if(rotateArray[mid] <= rotateArray[r]){
                r = mid;
            }
            else if(rotateArray[mid] >= rotateArray[l]){
                l = mid;
            }
        }
        
        return 0;
    }
};
```

## 矩阵中的路径

**描述**
```
请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。 例如 a b c e s f c s a d e e 这样的3 X 4 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。
```
- dfs+回溯
```c++
class Solution {
public:
    bool hasPath(char* matrix, int rows, int cols, char* str)
    {
        bool* visted = new bool[rows * cols]{false};
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++){
                if(matrix[i*cols + j] == str[0]){
                    // 注意这里是i*cols + j, 因为这个错误卡了很久
                    visted[i*cols + j] = true;
                    if(dfs(matrix, i, j, 0, rows, cols, visted, str)) 
                        return true;
                    visted[i*cols + j] = false;
                }
            }
        return false;
    }

    bool dfs(char* matrix, int i, int j, int index, int rows, int cols, bool* visted, char* str){
        if(index == (strlen(str) - 1))
            return true;
        
        int dir[4][2] = {{1,0}, {0,1}, {-1,0}, {0,-1}};
        for(int next = 0; next < 4; next++){
            int next_i = i + dir[next][0];
            int next_j = j + dir[next][1];
            if((next_i >= 0) && (next_i < rows) && (next_j >= 0) && (next_j < cols)){
                if((matrix[next_i*cols + next_j] == str[index + 1]) && (visted[next_i*cols + next_j] == false)){
                    visted[next_i*cols + next_j] = true;
                    if(dfs(matrix, next_i, next_j, index + 1, rows, cols, visted, str)) 
                        return true;
                    visted[next_i*cols + next_j] = false;
                }
            }
        }
        return false;
    }
};
```

## 整数拆分(动态规划|贪心)
> LeetCode/[整数拆分](https://leetcode-cn.com/problems/integer-break/submissions/)
> NowCoder/剪绳子

**描述**
```
给定一个正整数 n，将其拆分为至少两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。

示例 1:

输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1。
示例 2:

输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36。
说明: 你可以假设 n 不小于 2 且不大于 58。
```
**题解**
```
找规律:
f(2)=1*1=1
f(3)=1*2=2
f(4)=2*2=4
f(5)=2*3=6
f(6)=3*3=9
f(7)=2*2*3=12
注意到，不会出现大于3的因子，如4、5等，如果有4则可拆成2*2，有5则可拆成2*3，因此可以贪心去找最多的3，但不要出现1。
而动规的转移方程为dp[n]=max{dp[n-2]*2, dp[n-3]*3, 2*(n-2), 3*(n-3)}, 之所以后两项是因为dp[2]和dp[3]会出现1(至少要分成两个整数)，这不是我们想要的情况。
```
```c++
class Solution {
public:
    int integerBreak(int n) {
        vector<int> dp(n+1 , 0);
        dp[1] = 1;
        dp[2] = 1;
        dp[3] = 2;
        for(int i = 4; i <= n; i++){
            int temp1 = max(dp[i - 2] * 2, 2 * (i - 2));
            int temp2 = max(dp[i - 3] * 3, 3 * (i - 3));
            dp[i] = max(temp1, temp2);
        }
        return dp[n];
    }
};
```

## 二进制中1的个数（位运算）
> NowCoder/[二进制中1的个数](https://www.nowcoder.com/practice/8ee967e43c2c4ec193b040ea7fbb10b8?tpId=13&tqId=11164&tPage=1&rp=3&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)
- `n&n-1` 可以去除最后一位的1
- 这题方法有很多，参考：[算法-求二进制数中1的个数](https://www.cnblogs.com/graphics/archive/2010/06/21/1752421.html)
```c++
class Solution {
public:
     int  NumberOf1(int n) {
         int ret = 0;
         while(n){
             ret++;
             n = n & (n-1);
         }
         return ret;
     }
};
```

## 数值的整数次方（位运算）
> NowCoder/[数值的整数次方](https://www.nowcoder.com/practice/1a834e5e3e1a4b7ba251417554e07c00?tpId=13&tqId=11165&tPage=1&rp=3&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)

**描述**
```
给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
```
- 快速幂
- 时间复杂度O(logN)
```c++
class Solution {
public:
    double Power(double base, int exponent) {
        int p = abs(exponent);
        double ret = 1.0;
        while(p > 0){
            if(p & 1){  // 如果p是奇数
               ret *= base;
               p--;
            }
                
            base *= base; 
            p >>= 1;
        }
        return exponent > 0 ? ret : 1/ret;
    }
};
```

## 删除链表中重复的节点

**描述**
```
在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5
```
- 由于可能列表中全部都是重复结点，所以要设置一个哨兵节点
- 注意一定要利用`&&`的短路原理，先判定当前指针是否指向NULL
- 注意删除一个指针后要指向`NULL`
    - 原因是`delete`只释放指针指向的内存，但指针仍然指向这块内存
```c++
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* deleteDuplication(ListNode* pHead)
    {
        if(pHead == NULL) 
            return pHead;
        
        ListNode* head = new ListNode{-1};
        head->next = pHead;
        
        ListNode* pre = head;
        ListNode* cur = pHead;
        while(cur!=NULL && cur->next != NULL){
            if(cur->val != cur->next->val){
                pre = cur;
                cur = cur->next;
            }
            else{
                int tempVal = cur->val;
                while(cur != NULL && cur->val == tempVal){
                    auto temp = cur;
                    cur = cur->next;
                    
                    delete temp;
                    temp = NULL;
                }
                pre->next = cur;
            }
        }
        
        auto ret = head->next;
        delete head;
        head = NULL;
        return ret;
    }
};
```

## 调整数组顺序使奇数位于偶数前面（数组）

**描述**
```
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
```
- 由于需要相对位置不变，就是用类似冒泡排序的方法，每一趟保证提一个偶数到最后
- 如果不需要稳定的话，可使用快速排序的双指针法
```c++
class Solution {
public:
    void reOrderArray(vector<int> &array) {
         for(int i = 0; i < array.size()/2; i++)
             for(int j = 0; j < array.size() - i - 1; j++)
             {
                 if((array[j]%2 == 0) && (array[j+1]%2 != 0)){
                    swap(array[j], array[j+1]);
                 }
             }
    }
};
```

## 链表中倒数第k个结点

**描述**
```
输入一个链表，输出该链表中倒数第k个结点。
```
- 快慢指针
```c++
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        if(pListHead == NULL)
            return NULL;
        ListNode* fast = pListHead;
        ListNode* slow = pListHead;
        
        while(fast != NULL && k>0){
            fast = fast->next;
            k--;
        }
        
        if(k > 0)
            return NULL;
        
        while(fast != NULL){
            fast = fast->next;
            slow = slow->next;
        }
        return slow;
    }
};
```

## 链表中环的入口节点（链表）

**描述**
```
给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。
```
**思路**
- 设置快慢指针，快指针每次走两步，慢指针每次走一步，等它们相遇后。相遇后，将一指针指向头指针，然后两个指针同时向前（步长为一步），再次相遇即为入口。
> 证明参考:牛客网/[讨论区](https://www.nowcoder.com/questionTerminal/253d2c59ec3e4bc68da16833f79a38e4)
```c++
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead)
    {
        if(pHead == NULL)
            return NULL;
        ListNode* fast = pHead;
        ListNode* slow = pHead;
        while((slow->next != NULL) && (fast->next->next != NULL)){
            fast = fast->next->next;
            slow = slow->next;
            if(fast == slow){
                slow = pHead;
                while(slow != fast){
                    slow = slow->next;
                    fast = fast->next;
                } 
                return slow;
            }
        }
        return NULL;
    }
};
```

## 反转链表

**描述**
```
输入一个链表，反转链表后，输出新链表的表头。
```
- 三指针
```c++
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* ReverseList(ListNode* pHead) {
         if(pHead == nullptr) return pHead;
         ListNode* pre = pHead;
         ListNode* cur = pHead->next;
         ListNode* nex = cur->next;
         pre->next = nullptr;
         while(nex != nullptr)
         {
             cur->next = pre;
             pre = cur;
             cur = nex;
             nex = nex->next;
         }
         cur->next = pre;
         return cur;
    }
};
```

## 合并两个排序的链表
> Nowcoder/[合并两个排序的链表](https://www.nowcoder.com/practice/d8b6b4358f774294a89de2a6ac4d9337?tpId=13&tqId=11169&tPage=1&rp=3&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)   

**描述**
```
输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
```
- 递归
```c++
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
    {
        if(pHead1 == nullptr) return pHead2;
        if(pHead2 == nullptr) return pHead1;
        
        if(pHead1->val < pHead2->val)
        {
            pHead1->next = Merge(pHead1->next, pHead2);
            return pHead1;
        }
        else
        {
            pHead2->next = Merge(pHead1, pHead2->next);
            return pHead2;
        }
    }
};
```

## 二叉树的镜像
> Nowcoder/[二叉树的镜像](https://www.nowcoder.com/practice/564f4c26aa584921bc75623e48ca3011?tpId=13&tqId=11171&tPage=1&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking) 

**描述**
```
操作给定的二叉树，将其变换为源二叉树的镜像。
输入描述:
二叉树的镜像定义：源二叉树 
    	    8
    	   /  \
    	  6   10
    	 / \  / \
    	5  7 9 11
    	镜像二叉树
    	    8
    	   /  \
    	  10   6
    	 / \  / \
    	11 9 7  5
```

```c++
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    void Mirror(TreeNode *pRoot) {
        if(pRoot == NULL)
            return;
        swap(pRoot->left, pRoot->right);
        
        Mirror(pRoot->left);
        Mirror(pRoot->right);
    }
};
```

## 对称的二叉树
> NowCoder/[对称的二叉树](https://www.nowcoder.com/practice/ff05d44dfdb04e1d83bdbdab320efbcb?tpId=13&tqId=11211&tPage=3&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)  

**描述**
```
请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。
```
```c++
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    bool isSymmetrical(TreeNode* pRoot)
    {
        if(pRoot == NULL) return true;
        
        return dfs(pRoot->left, pRoot->right);
    }
    
    bool dfs(TreeNode* l, TreeNode* r){
        if(l == NULL && r == NULL) return true;
        if(l == NULL || r == NULL) return false;
        
        if(l->val == r->val){
            return dfs(l->left, r->right) && dfs(l->right, r->left);
        }
        else
            return false;
    }

};
```

## 顺时针打印矩阵
> NowCoder/[顺时针打印矩阵](https://www.nowcoder.com/practice/9b4c81a02cd34f76be2659fa0d54342a?tpId=13&tqId=11172&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

**描述**
```
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
```
```c++
class Solution {
public:
    vector<int> printMatrix(vector<vector<int> > matrix) {
        vector<int> ret;
        int rl = 0, rr = matrix.size() - 1;
        int cl = 0, cr = matrix[0].size() - 1;
        while(rl<=rr && cl<=cr){
            for(int i = cl; i <= cr; i++)
                ret.push_back(matrix[rl][i]);
            for(int i = rl + 1; i <= rr; i++)
                ret.push_back(matrix[i][cr]);
            if(rl != rr)  // 注意这里，要考虑总行数或总列数为奇数的情况
                for(int i = cr - 1; i >= cl; i--)
                    ret.push_back(matrix[rr][i]);
            if(cl != cr)
                for(int i = rr - 1; i > rl; i--)
                    ret.push_back(matrix[i][cl]);
            
            rl++; rr--;
            cl++; cr--;
        }
        return ret;
    }
};
```

## 包含min函数的栈（数据结构）
> NowCoder/[包含min函数的栈](https://www.nowcoder.com/practice/4c776177d2c04c2494f2555c9fcc1e49?tpId=13&tqId=11173&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

**描述**
```
定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））
```
- 使用辅助栈？（不明白这题考察什么）
```c++
class Solution {
    stack<int> s;
    stack<int> s_min;
public:
    void push(int value) {
        s.push(value);
        if(s_min.empty())     // 注意判空
            s_min.push(value);
        if(s_min.top() >= value) // 注意小于等于，否则pop会出错
            s_min.push(value);
    }
    void pop() {
        if(s.top() == s_min.top())
            s_min.pop();
        s.pop();
    }
    int top() {
        return s.top();
    }
    int min() {
        return s_min.top();
    }
};
```

## 栈的压入、弹出序列
> NowCode/[栈的压入、弹出序列](https://www.nowcoder.com/practice/d77d11405cc7470d82554cb392585106?tpId=13&tqId=11174&tPage=2&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)

**描述**
```
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
```
**思路**
- 定义一个辅助栈将入栈元素入栈
- 当栈顶与出栈数组的元素相符时，出栈（注意判空）
- 最后栈为空则出栈序列合法，反之则不合法
```c++
class Solution {
public:
    bool IsPopOrder(vector<int> pushV,vector<int> popV) {
        if(pushV.empty()) return false;
        
        stack<int> tmp;
        int j = 0;
        for(int i = 0; i < pushV.size(); i++){
            tmp.push(pushV[i]);
            while(!tmp.empty() && tmp.top() == popV[j]){ // 注意判空
                tmp.pop();
                j++;
            }
        }
        if(tmp.empty())
            return true;
        else
            return false;
    }
};
```

## 从上往下打印二叉树
> NowCode/[从上往下打印二叉树](https://www.nowcoder.com/practice/7fe2212963db4790b57431d9ed259701?tpId=13&tqId=11175&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

**描述**
```
从上往下打印出二叉树的每个节点，同层节点从左至右打印。
```
- 用队列来实现
```c++
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    vector<int> PrintFromTopToBottom(TreeNode* root) {
        vector<int> ret;
        if(root == NULL) 
            return ret;
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty()){
            auto temp = q.front();
            q.pop();
            ret.push_back(temp->val);
            if(temp->left != NULL)
                q.push(temp->left);
            if(temp->right != NULL)
                q.push(temp->right);
        }
        return ret;
    }
};
```

## 把二叉树打印成多行
> NowCode/[把二叉树打印成多行](https://www.nowcoder.com/practice/445c44d982d04483b04a54f298796288?tpId=13&tqId=11213&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

**描述**
```
从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行
```
- 多加入两个变量`curL`和`nexL`分别记录当前行/下一行要打印的数值，当`curL`为零时交换
- 利用额外的`temp`动态数组，每一行每一行地加入`ret`返回数组中
```c++
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
        vector<vector<int> > Print(TreeNode* pRoot) {
            vector<vector<int> > ret;
            vector<int> temp;
            if(pRoot == NULL) return ret;
            int curL = 1;
            int nexL = 0;
            queue<TreeNode* > q;
            q.push(pRoot);
            while(!q.empty()){
                auto cur  = q.front();
                q.pop();
                curL--;
                temp.push_back(cur->val);
                
                if(cur->left != NULL){
                    q.push(cur->left);
                    nexL++;
                }
                
                if(cur->right != NULL){
                    q.push(cur->right);
                    nexL++;
                }
                
                if(curL == 0){
                    ret.push_back(temp);
                    temp.clear();
                    curL = nexL;
                    nexL = 0;
                }
            }
            return ret;
        }
    
};
```

## 按之字形顺序打印二叉树
> NowCoder/[按之字形顺序打印二叉树](https://www.nowcoder.com/practice/91b69814117f4e8097390d107d2efbe0?tpId=13&tqId=11212&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

**描述**
```
请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。
```
- 按照上题代码加个`reverse`的奇偶判断
- 用栈的思路解决，定义两个栈，分别负责从左到右输出和从右往左输出。
```c++
class Solution {
public:
        vector<vector<int> > Print(TreeNode* pRoot) {
            vector<vector<int> > ret;
            vector<int> temp;
            if(pRoot == NULL) 
                return ret;
            int index = 0;
            
            // s[0]从左到右输出，s[1]从右往左输出
            // s[0]从右往左入栈，s[1]从左往右入栈
            stack<TreeNode* > s[2];
            s[0].push(pRoot);
            while(!s[index & 1].empty()){
                auto cur = s[index & 1].top();
                s[index & 1].pop();
                temp.push_back(cur->val);
                
                if(index & 1){// s[1]
                    if(cur->right != NULL)
                        s[0].push(cur->right);
                    
                    if(cur->left != NULL)
                        s[0].push(cur->left);
                }
                else{//s[0]
                    if(cur->left != NULL)
                        s[1].push(cur->left);
                    
                    if(cur->right != NULL)
                        s[1].push(cur->right);
                }
                
                if(s[index & 1].empty()){
                    index++;
                    ret.push_back(temp);
                    temp.clear();
                }
                
            }
            return ret;
        }
    
};
```

## 二叉搜索树的后序遍历序列
> NowCoder/[二叉搜索树的后序遍历序列](https://www.nowcoder.com/practice/a861533d45854474ac791d90e447bafd?tpId=13&tqId=11176&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

**描述**
```
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
```
- 二叉搜索树的定义: 左子树的值小于右子树
- 递归判断，定义`mid`分界点，注意考虑没有右子树的情况，即`mid`不存在时的对应取值
```c++
class Solution {
public:
    bool VerifySquenceOfBST(vector<int> sequence) {
        if(sequence.empty())
            return false;
        
        return dfs(sequence, 0, sequence.size());
    }
    
    bool dfs(vector<int>& s, int l, int r){
        if(r - l <= 1)
            return true;
        int base = s[r-1];
        int mid = l;
        for(;mid < r-1; mid++)
            if(s[mid] > base){
                break;
            }
            
        for(int i = mid; i < r-1; i++){
            if(s[i] < base)
                return false;
        }
        
        return dfs(s, l, mid) && dfs(s, mid, r-1);
    }
};
```

## 二叉搜索树的第k个结点
> NowCoder/[二叉搜索树的第k个结点](https://www.nowcoder.com/practice/ef068f602dde4d28aab2b210e859150a?tpId=13&tqId=11215&tPage=4&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)

**描述**
```
给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。
```
- 思路：二叉搜索树按照中序遍历的顺序打印出来正好就是排序好的顺序。所以，按照中序遍历顺序找到第k个结点就是结果。
```c++
class Solution {
    vector<TreeNode*> res;
public:
    TreeNode* KthNode(TreeNode* pRoot, int k)
    {    
        if(pRoot == nullptr || k <= 0)
            return nullptr;
        dfs(pRoot);
        return k <= res.size() ? res[k-1] : nullptr;
    }
    
    void dfs(TreeNode* t){
        if(t == nullptr) return;
        dfs(t->left);
        res.push_back(t);
        dfs(t->right);
    }
    
};
```
- 精简版
```c++
class Solution {
    int t = 0;
public:
    TreeNode* KthNode(TreeNode* pRoot, int k)
    {    
        if(pRoot){
            auto node = KthNode(pRoot->left, k);
            if(node) return node; // 如果返回值不为nullptr,说明已经找到第k小的结点，直接返回这个结点
            if(++t == k) return pRoot;
            node = KthNode(pRoot->right, k);
            if(node) return node;
        }
        return nullptr; // 注意正常中序遍历是没有返回值的，这里返回nullptr
    }
};
```

## 二叉树中和为某一值的路径
> NowCoder/[二叉树中和为某一值的路径](https://www.nowcoder.com/practice/b736e784e3e34731af99065031301bca?tpId=13&tqId=11177&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

**描述**
```
输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)
```
- 路径是指根节点到叶子节点的路径
- 回溯法
```c++
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    vector<vector<int> > ret;
    vector<int> trace;
    vector<vector<int> > FindPath(TreeNode* root,int expectNumber) {
        if(root != nullptr)
            dfs(root, expectNumber);
        return ret;
    }
    
    void dfs(TreeNode* cur, int n){
        trace.push_back(cur->val);
        if(cur->left == nullptr && cur->right == nullptr){
            if(cur->val == n){
                ret.push_back(trace);
            }
        }
        
        if(cur->left != nullptr){
            dfs(cur->left, n - cur->val);
        }
        if(cur->right != nullptr){
            dfs(cur->right, n - cur->val);
        }
        trace.pop_back();
    }
};
```

## 复杂链表的控制
> NowCoder/[复杂链表的控制](https://www.nowcoder.com/practice/b736e784e3e34731af99065031301bca?tpId=13&tqId=11177&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

**描述**
```
输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
```
- 问题的难点在于无法实时知道特殊指针指向的位置
- 因此需要先复制所有节点，再找到特殊指针指向的位置，再断掉指针
```c++
/*
struct RandomListNode {
    int label;
    struct RandomListNode *next, *random;
    RandomListNode(int x) :
            label(x), next(NULL), random(NULL) {
    }
};
*/
class Solution {
public:
    RandomListNode* Clone(RandomListNode* pHead)
    {
        if(pHead == nullptr)
            return nullptr;
        auto cur = pHead;
        while(cur != nullptr){
            auto t = new RandomListNode(cur->label);
            t->next = cur->next;
            cur->next = t;
            cur = t->next;
        }
        
        cur = pHead;
        while(cur != nullptr){
            auto t = cur->next;
            if(cur->random != nullptr)
                t->random = cur->random->next;
            cur = t->next;
        }
        
        auto ret = pHead->next;
        cur = pHead;
        while(cur->next != nullptr){ // 考虑组后只有两个节点的情况，断完后, cur->next==nullptr，循环不再继续
            auto t = cur->next;
            cur->next = t->next;
            cur = t;
        }
        return ret;
    }
};
```
## 平衡二叉树
> NowCoder/[平衡二叉树](https://www.nowcoder.com/practice/8b3b95850edb4115918ecebdf1b4d222?tpId=13&tqId=11192&tPage=2&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)

**描述**
```
输入一棵二叉树，判断该二叉树是否是平衡二叉树。
```
- 注意要从下往上遍历，防止重复遍历
```c++
class Solution {
public:
    bool IsBalanced_Solution(TreeNode* pRoot) {
         return Depth(pRoot) == -1 ? false : true;
    }
    
    int Depth(TreeNode* pRoot){
        if(pRoot == nullptr)
            return 0;
        
        int left, right;
        left = Depth(pRoot->left);
        if(left == -1)
            return -1;
        right = Depth(pRoot->right);
        if(right == -1)
            return -1;
        
        if(abs(right - left) > 1)
            return -1;
        else
            return max(right, left) + 1;
    }
};
```

## 二叉搜索树与双向链表
> NowCoder/[二叉搜索树与双向链表](https://www.nowcoder.com/practice/947f6eb80d944a84850b0538bf0ec3a5?tpId=13&tqId=11179&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

**描述**
```
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
```
- 中序遍历的顺序即为排序好的顺序
- 在中序遍历的基础上，添加双向指针即可
```c++
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    TreeNode* pre = nullptr;
    TreeNode* ret = nullptr; 
    
    TreeNode* Convert(TreeNode* pRootOfTree)
    {
        dfs(pRootOfTree);
        return ret;
    }
    
    void dfs(TreeNode* node){
        if(node == nullptr)
            return;
        dfs(node->left);
        
        // 找到头节点，只操作一次
        if(ret == nullptr)
            ret = node;
        
        // 添加双向指针
        if(pre != nullptr)
            pre->right = node;
        node->left = pre;
        pre = node;
        dfs(node->right);
    }
};
```

## 序列化二叉树
> NowCoder/[序列化二叉树](https://www.nowcoder.com/practice/cf7e25aa97c04cc1a68c8f040e71fb84?tpId=13&tqId=11214&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

**描述**
```
请实现两个函数，分别用来序列化和反序列化二叉树。
接口如下：
  char* Serialize(TreeNode *root);
  TreeNode* Deserialize(char *str);
```
```c++
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
    stringstream ss_serial;
    stringstream ss_deserial;
public:
    char* Serialize(TreeNode *root) {    
        dfs_serial(root);
        char result[1024];
        // str()用来取出字符串，c_str()用来转换成const* char;
        return strcpy(result, ss_serial.str().c_str());
    }
    
    void dfs_serial(TreeNode *root)
    {
        // 中序遍历
        if(root == nullptr) {ss_serial << "#"; return;}
        ss_serial << root->val ;
        ss_serial << ",";
        dfs_serial(root->left);
        ss_serial << ",";
        dfs_serial(root->right);
    }
    
    TreeNode* Deserialize(char *str) {
        if(strlen(str) < 1) return nullptr;
        
        ss_deserial << str;
        return dfs_deserial();    
    }
    
    TreeNode* dfs_deserial(){
        if(ss_deserial.eof()) return nullptr;
        
        string val;
        getline(ss_deserial, val, ',');

        if(val == "#") return nullptr;
        else
        {
            // stoi从字符转换成整数
            TreeNode* node = new TreeNode{ stoi(val) };
            node->left = dfs_deserial();
            node->right = dfs_deserial();
            return node;
        }
        
    }
};
```

## 字符串的排列
> NowCoder/[字符串的排列](https://www.nowcoder.com/practice/fe6b651b66ae47d7acce78ffdd9a96c7?tpId=13&tqId=11180&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

**描述**
```
输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
```
```c++
class Solution {
    vector<string> ret;
    string temp;
    vector<int> used;
    int length;

public:
    vector<string> Permutation(string str) {
        if(str.empty())
            return ret;
        sort(str.begin(), str.end());
        length = str.size();
        temp.resize(length, '\0');
        used.resize(length, 0);
        dfs(str, 0);
        return ret;
    }

    void dfs(const string& s, int step){
        if(step == length){
            ret.push_back(temp);
            return;
        }
        
        for(int i = 0; i < length; i++){
            if(used[i])
                continue;
            // 注意重复, 重复值在对应位置只使用一次
            if(!used[i - 1] && i > 0 && s[i-1] == s[i])
                continue;
            used[i] = 1;
            temp[step] = s[i];
            dfs(s, step + 1);
            used[i] = 0;
        }
    }
};
```

## 数组中超过一半的数字
> NowCoder/[数组中超过一半的数字](https://www.nowcoder.com/practice/e8a1b01a2df14cb2b228b30ee6a92163?tpId=13&tqId=11181&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

**描述**
```
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
```
- hash
- 多数投票算法
1. 如果count==0，则将majority的值设置为数组的当前元素 count++；
2. 如果count!=0，如果majority和现在数组元素值相同，则count++，反之count--；
3. 重复上述两步，直到扫描完数组。
4. count赋值为0，再次从头扫描数组，如果素组元素值与majority的值相同则count++，直到扫描完数组为止。
5. 如果此时count的值大于等于n/2，则返回majority的值，反之则返回-1。

```c++
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        unordered_map<int, int> m;
        int n = numbers.size() / 2;
        for(auto i : numbers){
            m[i]++;
            if(m[i] > n)
                return i;
        }
        return 0;
    }
};
```
```c++
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        int count = 0;
        int majority = 0;
        for(auto i : numbers){
            if(count == 0){
                majority = i;
                count++;
            }
            else{
                if(majority == i)
                    count++;
                else
                    count--;
            }
        }
        
        
        count = 0;
        for(auto i : numbers){
            if(i == majority)
                count++;
            if(count > numbers.size()/2)
                return majority;
        }
        return 0;
    }
};

```

## 数组中第k大的元素
> NowCoder/[最小的k个数](https://www.nowcoder.com/practice/6a296eb82cf844ca8539b57c23e6e9bf?tpId=13&tqId=11182&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
> LeetCode/[数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/description/)

- 快排
- 查找第K个为O(N)
- 如果还要排序O(N+klogk)
- In average, this algorithm reduces the size of the problem by approximately one half after each partition, giving the recurrence T(n) = T(n/2) + O(n) with O(n) being the time for partition. The solution is T(n) = O(n), which means we have found an average linear-time solution. However, in the worst case, the recurrence will become T(n) = T(n - 1) + O(n) and T(n) = O(n^2)
- 高频面试题
```c++
// NowCoder
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        if(k > input.size())
            return vector<int>();
        quicksort(input, 0, input.size() - 1, k);

        return vector<int>(input.begin(), input.begin() + k);
    }
    
    void quicksort(vector<int>& input, int l, int r, int k)
    {
        if(l >= r)
            return;
            
        int index = partition(input, l, r);
        
        if(index >= k)
            quicksort(input, l, index - 1, k);
        else{
            quicksort(input, l, index - 1, k);
            quicksort(input, index + 1, r, k);
        }
    }
    
    int partition(vector<int>& input, int l, int r){
        int i = l;
        int base = rand() % (r - l + 1) + l; // rand返回的是[0, 最大随机数]
        int pivot  = input[base];
        swap(input[base], input[r]);
        for(int j = l; j < r; j++){
            if(input[j] < pivot){
                swap(input[i], input[j]);
                i++;
            }
        }
        swap(input[r], input[i]);
        return i;
    }
};
```
```c++
// LeetCode
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        // if(k > nums.size())
        //     return 0;
        quicksort(nums, 0, nums.size() - 1, k);
        return nums[k-1];
    }
    
    void quicksort(vector<int>& input, int l, int r, int k)
    {
        if(l >= r)
            return;
            
        int index = partition(input, l, r);
        if(index == k-1)
            return;
        
        if(index >= k)
            quicksort(input, l, index - 1, k);
        else{
            quicksort(input, l, index - 1, k);
            quicksort(input, index + 1, r, k);
        }
    }
    
    int partition(vector<int>& input, int l, int r){
        int i = l;
        int base = rand() % (r - l + 1) + l; 
        int pivot  = input[base];
        swap(input[base], input[r]);
        for(int j = l; j < r; j++){
            if(input[j] > pivot){
                swap(input[i], input[j]);
                i++;
            }
        }
        swap(input[r], input[i]);
        return i;
    }
};
```
- 堆
- 直接根据a[0...k]建堆，时间复杂性为O(k)。遍历a[0...n-1]的时间复杂性为O(n)。找到比堆顶元素小的数后，进堆的时间复杂性为O(klogk)，出堆的时间复杂性为O(klogk)。输出有序的前k小的数的时间复杂性为O(klogk)，输出无序的前k小的数的时间复杂性为O(k)
- 综合时间复杂度为O(N+klogk)
- Nowcoder/小顶堆不优化
```c++
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        if(k > input.size() || k <= 0 || input.empty())
            return vector<int>();
        if(k == input.size())
            return input;
        
        priority_queue<int, vector<int>, greater<int> > q; 

        for(auto i : input)
            q.push(i);
        
        vector<int> ret;
        for(int i = 0; i < k; i++){
            ret.push_back(q.top());
            q.pop();
        }
        
        return ret;
    }
};
```

- Nowcoder/大顶堆优化
```c++
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        if(k > input.size() || k <= 0 || input.empty())
            return vector<int>();
        if(k == input.size())
            return input;
        
        priority_queue<int, vector<int> > q; // 大顶堆

        for(int i = 0; i < input.size(); i++){
            if(i >= k){ // 注意这里要加花括号，不然有歧义
                if(input[i] < q.top()){
                    q.pop();
                    q.push(input[i]);
                }
               }
            else
                q.push(input[i]);
        }
        
        vector<int> ret;
        for(int i = 0; i < k; i++){
            ret.push_back(q.top());
            q.pop();
        }

        return ret;
    }
};
```

## 数据流中的中位数
> NowCoder/[数据流中的中位数](https://www.nowcoder.com/practice/9be0172896bd43948f8a32fb954e1be1?tpId=13&tqId=11216&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

**描述**
```
如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。
```
- 使用两个堆，一个大顶堆，一个小顶堆，大顶堆永远小于小顶堆，保持平衡，奇数个就放在大顶堆中多放一个。
```c++
class Solution {
    priority_queue<int> big; 
    priority_queue<int, vector<int>, greater<int> > small;
    int N;
public:
    void Insert(int num)
    {
        N++;
        if(N & 1){
            small.push(num);
            auto t = small.top();
            small.pop();
            big.push(t);
        }
        else{
            big.push(num);
            auto t = big.top();
            big.pop();
            small.push(t);
        }
    }

    double GetMedian()
    { 
        if(N & 1)
            return (double)big.top();
        else
            return (double)(small.top() + big.top())/2;
    }

};
```

## 数字1的个数
> NowCoder/[数字1的个数](https://www.nowcoder.com/practice/bd7f978302044eee894445e244c7eee6?tpId=13&tqId=11184&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
> LeetCode/[数字1的个数](https://leetcode-cn.com/problems/number-of-digit-one/submissions/)
**描述**
```
求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。
```
- 数学找规律题
- 分位数讨论，某位上的1出现了多少次，比如90，则十位上的1出现了1次，个位上的出现了9次
> LeetCode/[讨论区]/(https://leetcode.com/problems/number-of-digit-one/discuss/64381/4%2B-lines-O(log-n)-C%2B%2BJavaPython)

```c++
class Solution {
public:
    int countDigitOne(int n) {
        int ret = 0;
        for(long long m = 1; m <= n; m*=10){
            auto a = n / m;
            auto b = n % m;
            ret += (a + 8) / 10 * m + (a % 10 == 1)*(b + 1); 
        }
        return ret;
    }
};
```

## 连续子数组的最大和
> NowCoder/[连续子数组的最大和](https://www.nowcoder.com/practice/459bd355da1549fa8a49e350bf3df484?tpId=13&tqId=11183&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

**描述**
```
{6,-3,-2,7,-15,1,2,2}，连续子数组的最大和为 8（从第 0 个开始，到第 3 个为止）
```
- O(N)算法，如果前缀和大于0则列入，否则丢弃
- 注意有全负序列，初始化最好用第一个元素
```
class Solution {
public:
    int FindGreatestSumOfSubArray(vector<int> array) {
        if(array.size() == 0) 
            return int();
        
        int sum = array[0];
        int ret = array[0];
        for(int i = 1; i < array.size(); i++){
            if(sum < 0)
                sum = array[i];
            else
                sum += array[i];
            
            if(sum > ret)
                ret = sum;
        }
        
        return ret;
    }
};
```


## 把数组排成最小的数
> NowCoder/[把数组排成最小的数](https://www.nowcoder.com/practice/8fecd3f8ba334add803bf2a06af1b993?tpId=13&tqId=11185&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

**描述**
```
输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
```
- 运用排序算法，定义`cmp`
- c++中将数字转换成字符串
  - `to_string()`
```c++
class Solution {
public:
    string PrintMinNumber(vector<int> numbers) {
        sort(numbers.begin(), numbers.end(), [](const int& l, const int& r){
           return to_string(l) + to_string(r) < to_string(r) + to_string(l);             
        });
        
        stringstream ss;
        for(auto i : numbers)
            ss<<to_string(i);
        return ss.str();
    }
};
```

## 把数字翻译成字符串（解码方法）
> LeetCode/[解码方法](https://leetcode-cn.com/problems/decode-ways/description/)
> 题解：bilibili/[花花酱](https://www.bilibili.com/video/av42112212)

**描述**
```
一条包含字母 A-Z 的消息通过以下方式进行了编码：

'A' -> 1
'B' -> 2
...
'Z' -> 26
给定一个只包含数字的非空字符串，请计算解码方法的总数。

示例 1:

输入: "12"
输出: 2
解释: 它可以解码为 "AB"（1 2）或者 "L"（12）。
```
- 动态规划
- dp[i]: ways to decode s[0]....s[i]
- dp[i] = dp[i-1] if(s[i]合法) else 0 + dp[i-1] if(s[i-1, i]合法) else 0 
- 另外，如果都不合法，则直接返回0
- 其中一位数如果为0不合法，其他合法， 两位数必须在[10，26]范围内才合法；
```c++
class Solution {
public:
    int numDecodings(string s) {
        int n = s.size();
        if(n <= 0 || s[0] == '0') 
            return 0;
        if(n == 1) 
            return 1;

        int prefix1;
        int prefix2 = (s[0] - '0')*10 + (s[1] - '0'); 
        vector<int> dp(n, 0);
        dp[0] = isValid_1(s[0] - '0');
        dp[1] = isValid_1(s[1] - '0') + isValid_2(prefix2);
        
        if(dp[1] == 0)
            return 0;
        for(int i = 2; i < n; i++){
            prefix1 = s[i] - '0';
            prefix2 = (s[i - 1] - '0')*10 + (s[i] - '0'); 
            dp[i] += (isValid_1(prefix1)) * dp[i - 1];
            dp[i] += (isValid_2(prefix2)) * dp[i - 2];
            if(dp[i] == 0)
                return 0;
        }
        return dp[n - 1];
    }
    
    int isValid_1(int n){
        if(n == 0)
            return 0;
        else
            return 1;
    }
    int isValid_2(int n){
        if(n >= 10 && n <= 26)
            return 1;
        else
            return 0;
    }
    
};
```

## 礼物的最大价值
> NowCoder/[牛客网](https://www.nowcoder.com/questionTerminal/72a99e28381a407991f2c96d8cb238ab)

**描述**
```
在一个 m*n 的棋盘的每一个格都放有一个礼物，每个礼物都有一定价值（大于 0）。
从左上角开始拿礼物，每次向右或向下移动一格，直到右下角结束。
给定一个棋盘，求拿到礼物的最大价值。例如，对于如下棋盘
    1    10   3    8
    12   2    9    6
    5    7    4    11
    3    7    16   5
礼物的最大价值为 1+12+5+7+7+16+5=53。
```
- dp[i, j]代表经过[i, j]的礼物最大值
- dp[i, j] = max(dp[i-1, j], dp[i, j-1]) + board[i, j]
```c++
class Bonus {
public:
    int getMost(vector<vector<int> > board) {
        if(board[0].size()<=0 || board.size() <= 0)
            return 0;
        int n = board.size();
        int m = board[0].size();
        
        vector<vector<int> > dp(n, vector<int>(m, 0));
        
        for(int i = 0; i < n; i++)
            for(int j = 0; j < m; j++){
                int a = 0;
                int b = 0;
                if(i - 1 >= 0)
                    a = dp[i - 1][j];
                if(j - 1 >= 0)
                    b = dp[i][j - 1];
                dp[i][j] = max(a, b) + board[i][j];
            }
        return dp[n - 1][m - 1];
    }
};
```

## 丑数
> LeetCode/[丑数](https://leetcode-cn.com/problems/ugly-number/)

**描述**
```
编写一个程序判断给定的数是否为丑数
```
- 注意边界判断，尤其num=0的情况
```c++
class Solution {
public:
    bool isUgly(int num) {
        if(num <= 0)
            return false;
        while(num%2 == 0) num /= 2;
        while(num%3 == 0) num /= 3;
        while(num%5 == 0) num /= 5;
        return num==1;
    }
};
```


## 丑数II
> NowCoder/[丑数](https://www.nowcoder.com/practice/6aa9e04fc3794f68acf8778237ba065b?tpId=13&tqId=11186&tPage=2&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)  
> LeetCode/[丑数II](https://leetcode-cn.com/problems/ugly-number-ii/submissions/)  
> bilibili/[花花酱](https://www.bilibili.com/video/av31571137)

**描述**
```
把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数
```
- 三指针记录上一次乘以对应质因子的位置，然后排序
- c++11中可以使用初始化列表传入`min`函数
  - `min({x, y, z})`
```c++
class Solution {
public:
    int nthUglyNumber(int n) {
        vector<int> nums{1};
        int index2 = 0;
        int index3 = 0;
        int index5 = 0;
        for(int i = 1; i < n; i++){
            int next2 = 2 * nums[index2];
            int next3 = 3 * nums[index3];
            int next5 = 5 * nums[index5];
            
            nums.push_back(min({next2, next3, next5}));
            if(nums[i] == next2)
                index2++;
            if(nums[i] == next3)
                index3++;
            if(nums[i] == next5)
                index5++;
        }
        return nums[n-1];
    }
};

```

## 第一个只出现一次的字符
> NowCoder/[第一个只出现一次的字符](https://www.nowcoder.com/practice/1c82e8cf713b4bbeb2a5b31cf5b0417c?tpId=13&tqId=11187&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

**描述**
```
在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写)
```
- Hash
- 判断一个元素是否字hash表中
  - `m.find(i) == m.end()`
    - find返回迭代值
    - 返回值->first 是key, 返回值->second 是value
  - `m.count(i) == 0`
```c++
class Solution {
public:
    int FirstNotRepeatingChar(string str) {
        unordered_map<char, int> m;
        for(auto i : str){
            if(m.find(i) == m.end())
                m[i] = 1;
            else
                m[i]++;
        }
        
        for(int i = 0; i < str.size(); i++)
            if(m[str[i]] == 1)
                return i;
        
        return -1;
    }
};
```
## 字符流中第一个只出现一次的字符
> NowCoder/[字符流中第一个只出现一次的字符](https://www.nowcoder.com/practice/00de97733b8e4f97a3fb5c680ee10720?tpId=13&tqId=11207&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
**描述**
```
请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。
```
- hash
- 利用队列，每次插入时入队
  - 当队首元素已经重复出现时，弹出队首
  - 则队首元素就是第一个只出现一次的字符
```c++
class Solution
{
public:
    unordered_map<char, int> m;
    queue<char> q;
    void Insert(char ch)
    {
        if(m.count(ch) == 0)
            m[ch] = 1;
        else
            m[ch]++;
        q.push(ch);
        while(!q.empty() && m[q.front()] > 1)
            q.pop();
    }
  //return the first appearence once char in current stringstream
    char FirstAppearingOnce()
    {
        return q.empty() ? '#' : q.front();
    }

};
```

## 替换空格
> NowCoder/[替换空格](https://www.nowcoder.com/practice/4060ac7e3e404ad1a894ef3e17650423?tpId=13&tqId=11155&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

**描述**
```
请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
```
```c++
class Solution {
public:
	void replaceSpace(char *str,int length) {
        if(str == nullptr || length < 0)
            return;
        unsigned int t = 0;
        for(int i = 0; i < length; i++){
            if(str[i] == ' ')
                t++;
        }
        
        t *= 2;
        int n = length + t;
        str[n--] = '\0';
        for(int i = length - 1; i >= 0; i--){
            if(str[i] == ' '){
                str[n--] = '0';
                str[n--] = '2';
                str[n--] = '%';
            }
            else
                str[n--] = str[i];
        }
	}
};
```


## 翻转单词序列
> NowCoder/[翻转单词序列](https://www.nowcoder.com/practice/3194a4f4cf814f63919d0790578d51f3?tpId=13&tqId=11197&tPage=3&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)

**描述**
```
牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？
```
```c++
class Solution {
public:
    string ReverseSentence(string str) {
        string res="", temp="";
        for(int i = 0; i < str.size(); i++){
            if(str[i] == ' ')
                res = " " + temp + res, temp = ""; 
            else
                temp += str[i];
        }
        if(temp.size())
            res = temp + res;
        return res;
    }
};
```

## 不用加减乘除做加法
> NowCoder/[不用加减乘除做加法](https://www.nowcoder.com/practice/59ac416b4b944300b617d4f7f111b215?tpId=13&tqId=11201&tPage=3&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)

- 位运算
- 先用`^`找到去除进位的sum值
- 再用`&`找到进位
- 不停循环
```c++
class Solution {
public:
    int Add(int num1, int num2)
    {
        while(num2 != 0){
            int sum = num1 ^ num2;
            int carry = (num1 & num2) << 1;
            num1 = sum;
            num2 = carry;
        }
        return num1;
    }
};
```

## 扑克牌顺子
> NowCoder/[扑克牌顺子](https://www.nowcoder.com/practice/762836f4d43d43ca9deb273b3de8e1f4?tpId=13&tqId=11198&tPage=3&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking)

**描述**
```
LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。
```
- hash判断对子
- 记录除大王和小王的差，如果差小于5，则返回true
```c++
class Solution {
public:
    bool IsContinuous( vector<int> numbers ) {
        if(numbers.size() != 5)
            return false;
        int maxN = -1;
        int minN = 14;
        int m[14]={0};
        for(int i = 0; i < 5; i++){
            if(numbers[i] == 0) continue;
            m[numbers[i]]++;
            if(m[numbers[i]] >= 2) return false;
            maxN = max(maxN, numbers[i]);
            minN = min(minN, numbers[i]);
        } 
        if(maxN - minN < 5)
            return true;
        return false;
    }
};
```