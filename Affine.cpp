#include <bits/stdc++.h>
using namespace std;
void minpenalty(string s1, string s2, int extension, int gap)
{
	int i, j;
	int m = s1.length(); 
	int n = s2.length(); 
 	int mat[m+n+1][m+n+1] = {0};
 	for (i = 0; i <= (n+m); i++)
 	{
	  mat[i][0] = i * gap;
	  mat[0][i] = i * gap;
 	}
 	for (i = 1; i <= m; i++)
 	{
  		for (j = 1; j <= n; j++)
  		{
		   if (s1[i - 1] == s2[j - 1])
		   {
		    	mat[i][j] = mat[i - 1][j - 1] + extension;
		   }
		   else
   		   {
			mat[i][j] = min({mat[i - 1][j - 1] - gap  , mat[i - 1][j] - gap , mat[i][j - 1] -  gap });
   		   }
  		}
 	}
 	int l = n + m; 
 	i = m; j = n;
	int xc = l;
	int yc = l;
 	int resx[l+1], resy[l+1];
 	while ( !(i == 0 || j == 0))
 	{
		  if (s1[i - 1] == s2[j - 1])
		  {
		   resx[xc--] = (int)s1[i - 1];
		   resy[yc--] = (int)s2[j - 1];
		   i-=1;
		   j-=1;
		  }
		  else if (mat[i - 1][j - 1] + extension == mat[i][j])
		  {
		   resx[xc--] = (int)s1[i - 1];
		   resy[yc--] = (int)s2[j - 1];
		   i-=1;
		   j-=1;
		  }
		  else if (mat[i][j - 1] - gap == mat[i][j])
		  {
		   resx[xc--] = 42;
		   resy[yc--] = (int)s2[j - 1];
		   j-=1;
		  }
		    else if (mat[i - 1][j] - gap == mat[i][j])
		  {
		   resx[xc--] = (int)s1[i - 1];
		   resy[yc--] = 42;
		   i-=1;
		  }
 	}
 	while (xc > 0)
 	{
	  if (i > 0)
	  {
	  	resx[xc--] = (int)s1[--i];
	  }
	  else 
	  {
	  resx[xc--] = 42;
	  }
 	}
	 while (yc > 0)
	 {
	  if (j > 0) resy[yc--] = (int)s2[--j];
	  else resy[yc--] = 42;
	 }
	 int q = 1;
	 for (i = l; i >= 1; i--)
	 {
	  if ((resy[i] == 42 && resx[i] == 42))
	  {
	   q = i + 1;
	   break;
	  }
	 }
	 cout << "Minimum Penalty ";
	 cout << mat[m][n] << endl;
	 for (i = q; i <= l; i++)
	 {
	  cout<<(char)resx[i];
	 }
	 cout << "\n";
	 for (i = q; i <= l; i++)
	 {
	  	cout << (char)resy[i];
	 }
	 return;
}
int main()
{
	string s1 ;
	string s2 ;
	int mismatch , gap;
	cout << "String 1";
	cin >> s1;
	cout << "String 2";
	cin >> s2;
	cout << "MisMatch Penalty";
	cin >> mismatch;
	cout << "Gap Penalty";
	cin >> gap;
	minpenalty(s1, s2,mismatch, gap);
}
