入力1次元のベイズ最適化のサンプルコードです。今後多次元に対応予定です。
ベイズ最適化をとりあえず使ってみたい人向けのコードになります。

## 目次
* [ベイズ最適化とは](#ベイズ最適化とは)
* [何がすごいのベイズ最適化](#何がすごいのベイズ最適化)
* [本コードの使い方](#本コードの使い方)
* [解説](#解説)
* [詳しい解説](#詳しい解説)
* [License](#license)

## ベイズ最適化とは？
ベイズ最適化とはブラックボックス関数の最大・最小値を少ない回数で探索する手法になります。ガウス過程回帰などにより得た関数予測をもとに最適値を示す可能性の高いパラメータを提案するということを繰り返します。



## 何がすごいのベイズ最適化
ベイズ最適化の肝はガウス過程回帰にあります（個人の意見です）。ガウス過程回帰は普通の回帰と違い予測とその**不確かさ**を与えます。つまり、

**「この辺調べてないから不安だな」「この辺はたくさん調べたから自信あるな」**

という予測をするのです。これをもとに次のパラメータを決定するので

**「不安だから未知点を探索しよう（探索）」「不安な点があまりないから今の最大値付近が最大値なはずだ（活用）」**

などと提案してくれます。これにより普通の最適化手法よりも早く最適値を求めることができると言われています。現在では機械学習のハイパーパラメータの最適化や実験装置に組み込んで自律実験をすることなどに用いられています。

![image](なんとか.png)

## 本コードの使い方
まずは以下のパッケージをインストールします

```
pip install git+https://github.com/discood/bayesian-optimization.git
```

次に以下のパッケージをインストールします
* Matplotlib
* Numpy
* Scikit-learn

あとはbayesian_1obj.pyを実行するだけです

## 解説
今回のベイズ最適化の目標は f(x)=xsin(x) (0<=x<=2π)の最大化です。

1. data_num分ランダムに初期点が選択され、(x,y)がデータセットに追加されます。 

1. 次にデータセットを基にガウス過程回帰を用いて関数予測が行われます。

1. 次に関数予測をもとに次の探索すべきパラメータが提案されます。提案パラメータは獲得関数というものを最大にするxになります。

1. 提案されたパラメータxとそれに対応するf(x)がデータセットに追加されます。2.に戻ります。

このループを繰り返すことでf(x)を最大にするxが自動的に求められます。

loop数はloopで指定できます。また、loopが終了するとその時点での関数予測がgraphフォルダに出力されます。


## 詳しい解説
kernelはC*[RBF](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html "RBF kernel")になります。これらの中の変数は周辺尤度を最大化することで最適化されます。

獲得関数は最もシンプルな獲得関数の一つであるUCBを使用しています。

UCB(x) = μ + c * σ

こちらのcを大きくすると不確かさが大きい部分の探索が促されます。小さくすると現段階で大きいと考えられる点が探索されます。もっと色んな点を探索してほしいときはcをあげ、探索がなかなか収束しないときはcをさげるとよいでしょう。

## License
MIT License
