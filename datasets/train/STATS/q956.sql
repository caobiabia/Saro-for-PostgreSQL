select  count(*) from comments as c,  		posts as p,          users as u where c.UserId = u.Id 	and u.Id = p.OwnerUserId  AND p.PostTypeId=2  AND p.FavoriteCount<=8  AND u.Reputation>=1;
