select  count(*) from comments as c,  		posts as p,          users as u where c.UserId = u.Id 	and u.Id = p.OwnerUserId  AND p.Score>=0  AND p.Score<=17  AND p.ViewCount>=0  AND p.FavoriteCount>=0  AND u.Reputation>=1  AND u.Reputation<=304  AND u.Views>=0  AND u.Views<=108  AND u.UpVotes>=0  AND u.UpVotes<=10;