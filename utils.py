fn1 = '/Volumes/Transcend/text/68_linkedin.txt'
fn2 = '/Volumes/Transcend/text/rockyou-train.txt'

charset1 = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
charset2 = ['1','2','3','4','5','6','7','8','9','0']
charset3 = ['~','!','@','#','$','%','^','&','*','(',')','_','+','{','}','[',']',',','.','?','/',';','=','-','|',':','<','>']
charset4 = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def getclass(f):
	with open(f, 'r', encoding='ISO-8859-1') as f:
		for l in f.readlines():
			cnt = {'1':0, '2':0, '3':0, '4':0}
			if is_ascii(l):
				for w in l:
					if w in charset1:
						cnt['1'] += 1
					if w in charset2:
						cnt['2'] += 1
					if w in charset3:
						cnt['3'] += 1
					if w in charset4:
						cnt['4'] += 1
				for x in list(cnt.keys()):
					if cnt[x] == 0:
						del cnt[x]
				if len(cnt.keys()) == 1 and len(l)>=8:
					with open('1class8.txt', 'a') as f2:
						f2.write(l)
				if len(cnt.keys()) == 1 and len(l)>=16:
					with open('1class816.txt', 'a') as f2:
						f2.write(l)
				if len(cnt.keys()) == 3 and len(l)>=12:
					with open('3class12.txt', 'a') as f2:
						f2.write(l)
				if len(cnt.keys()) == 4 and len(l)>=8:
					with open('4class8.txt', 'a') as f2:
						f2.write(l)

getclass(fn1)
getclass(fn2)