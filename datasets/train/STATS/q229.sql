select  count(*) from posts as p,  		postLinks as pl,          postHistory as ph,          votes as v,          badges as b,          users as u  where p.Id = pl.RelatedPostId 	and u.Id = p.OwnerUserId 	and u.Id = b.UserId 	and u.Id = ph.UserId     and u.Id = v.UserId  AND b.Date<='2014-09-04 04:16:48'::timestamp  AND p.PostTypeId=1  AND p.ViewCount<=3169  AND p.CommentCount>=0  AND p.CommentCount<=6  AND p.FavoriteCount>=0  AND p.FavoriteCount<=17  AND u.Views>=0  AND u.Views<=4  AND u.DownVotes>=0  AND u.DownVotes<=1  AND v.VoteTypeId=2;