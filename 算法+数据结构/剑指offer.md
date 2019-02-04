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
**题目描述**

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
而动规的转移方程为dp[n]=max{dp[n-2]*2, dp[n-3]*3, 2*(n-2), 3*(n-3)}, 之所以后两项是因为dp[2]和dp[3]会出现1，这不是我们想要的情况。
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

