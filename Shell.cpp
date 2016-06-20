//#include <iostream>
//#include <vector>
//using namespace std;
////readCommand
////readArgs
//
//char* charVecToStr(vector<char> input)
//{
//	int i;
//	int size = input.size() + 1;
//	char *res = (char *)malloc(sizeof(char) * size);
//	for (i = 0; i < size - 1; i++) res[i] = input[i];
//	res[size - 1] = 0;
//	return res;
//}
//
//extern "C"
//void fetchTheCommand()
//{
//	char curChar;
//	char *str;
//	cin.clear();
//	printf_s(":");	
//	vector<char> command;
//	for (curChar = (char)cin.get(); (curChar != ' ') && (curChar != '\n'); curChar = (char)cin.get())
//	{
//		command.push_back(curChar);
//	}
//	str = charVecToStr(command);
//	printf("%s\n", str);
//	free(str);
//	command.clear();
//}
