class NormStandardScaler():
    def fit_tensor(self, x):
        # x = self.__standardize(x)
        # x = self.__normalize(x)  
        return x      

    def rescale(self, x):
        # x *= self.max - self.min
        # x += self.min

        # x *= self.std
        # x += self.mean

        return x

    def __normalize(self, x):
        self.min = x.min()
        self.max = x.max()
        x -= self.min # bring the lower range to 0
        x /= self.max - self.min # bring the upper range to 1    

        return x

    def __standardize(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)
        x -= self.mean
        x /= self.std
        return x