import Algorithm.Search (aStar)
import qualified Data.Map as Map
import System.Environment


type QubitMapping = Map.Map Int Int
type Swap = (Int, Int)

applySwap :: Swap -> QubitMapping -> QubitMapping
applySwap (u,v) = Map.map update
    where update x | x == u = v
                   | x == v = u
                   | otherwise = x

neighbors :: [Swap] -> QubitMapping -> [QubitMapping]
neighbors swaps mapping = [applySwap swap mapping | swap <- swaps]

distanceHeuristic :: Map.Map (Int, Int) Int -> QubitMapping -> QubitMapping -> Double
distanceHeuristic distances target current = fromIntegral(Map.foldrWithKey dist 0 current)/2
    where dist k val acc = acc + distances Map.! (target Map.! k, val)


findDist :: Map.Map (Int, Int) Int -> QubitMapping -> QubitMapping -> Maybe (Double, [QubitMapping])
findDist distances initial final = aStar (neighbors edges) (\x y -> 1) (distanceHeuristic distances final) (`Map.isSubmapOf` final)  initial
    where edges = [(u,v) | ((u, v), d)  <- filter ((== 1).snd) (Map.toList distances) ]

main = 
    do 
        [fname] <- getArgs
        [initial', final', distances'] <- lines <$> readFile fname 
        let initial = (Map.fromList $ read initial') :: QubitMapping
        let final = (Map.fromList $ read final') :: QubitMapping
        let distances = Map.fromList  (read distances' :: [((Int, Int), Int)])
        print $ findDist distances initial final
    